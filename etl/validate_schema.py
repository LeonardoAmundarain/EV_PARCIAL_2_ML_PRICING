"""
etl/validate_schema.py
Validación de esquema del dataset CSV antes de cargarlo a MongoDB.
Cumple con el requisito de "validación de esquemas" del pipeline ETL.
"""

import pandas as pd
import logging
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("etl.validate_schema")

# ── Esquema esperado del dataset ──────────────────────────────────────────────
COLUMNAS_OBLIGATORIAS = [
    "model_name", "model_slug", "provider", "provider_slug", "aa_id",
    "aa_intelligence_index", "pricing_tier", "blended_cost_usd_per_1m",
    "is_open_source", "release_year",
]

COLUMNAS_TIPO_NUMERICO = [
    "aa_intelligence_index", "aa_coding_index", "aa_math_index",
    "input_cost_usd_per_1m", "output_cost_usd_per_1m", "blended_cost_usd_per_1m",
    "output_tokens_per_second", "time_to_first_token_s", "release_year",
]

TIERS_VALIDOS = {"Free", "Budget", "Mid", "Premium", "Ultra", "Unknown"}

RANGO_INTELIGENCIA = (0, 100)
RANGO_ANIO = (2020, 2027)


@dataclass
class ResultadoValidacion:
    es_valido: bool
    errores: list = field(default_factory=list)
    advertencias: list = field(default_factory=list)
    filas_originales: int = 0
    filas_descartadas: int = 0

    def resumen(self) -> str:
        estado = "✓ VÁLIDO" if self.es_valido else "✗ INVÁLIDO"
        partes = [f"{estado} | filas: {self.filas_originales} | descartadas: {self.filas_descartadas}"]
        if self.errores:
            partes.append("ERRORES:\n  - " + "\n  - ".join(self.errores))
        if self.advertencias:
            partes.append("ADVERTENCIAS:\n  - " + "\n  - ".join(self.advertencias))
        return "\n".join(partes)


def validar_columnas_presentes(df: pd.DataFrame, resultado: ResultadoValidacion) -> bool:
    """Verifica que todas las columnas obligatorias existan en el DataFrame."""
    faltantes = [c for c in COLUMNAS_OBLIGATORIAS if c not in df.columns]
    if faltantes:
        resultado.errores.append(f"Columnas obligatorias ausentes: {faltantes}")
        return False
    return True


def validar_tipos_numericos(df: pd.DataFrame, resultado: ResultadoValidacion) -> pd.DataFrame:
    """
    Intenta convertir columnas numéricas; si una celda no es convertible
    se transforma en NaN y se reporta como advertencia (no detiene el pipeline).
    """
    for col in COLUMNAS_TIPO_NUMERICO:
        if col not in df.columns:
            continue
        antes_validos = df[col].notna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        despues_validos = df[col].notna().sum()
        perdidos = antes_validos - despues_validos
        if perdidos > 0:
            resultado.advertencias.append(
                f"Columna '{col}': {perdidos} valores no numéricos convertidos a nulo"
            )
    return df


def validar_pricing_tier(df: pd.DataFrame, resultado: ResultadoValidacion) -> pd.DataFrame:
    """Verifica que pricing_tier solo contenga valores del conjunto permitido."""
    if "pricing_tier" not in df.columns:
        return df
    invalidos = set(df["pricing_tier"].dropna().unique()) - TIERS_VALIDOS
    if invalidos:
        resultado.advertencias.append(
            f"Valores de pricing_tier fuera del esquema: {invalidos} — se normalizan a 'Unknown'"
        )
        df.loc[~df["pricing_tier"].isin(TIERS_VALIDOS), "pricing_tier"] = "Unknown"
    return df


def validar_rangos(df: pd.DataFrame, resultado: ResultadoValidacion) -> pd.DataFrame:
    """Valida que los valores numéricos estén dentro de rangos lógicos de negocio."""
    if "aa_intelligence_index" in df.columns:
        fuera_rango = ~df["aa_intelligence_index"].between(*RANGO_INTELIGENCIA) & df["aa_intelligence_index"].notna()
        n = fuera_rango.sum()
        if n > 0:
            resultado.advertencias.append(
                f"{n} registros con aa_intelligence_index fuera de rango {RANGO_INTELIGENCIA}, se descartan"
            )
            df = df[~fuera_rango]

    if "release_year" in df.columns:
        fuera_rango = ~df["release_year"].between(*RANGO_ANIO) & df["release_year"].notna()
        n = fuera_rango.sum()
        if n > 0:
            resultado.advertencias.append(
                f"{n} registros con release_year fuera de rango {RANGO_ANIO} corregidos a nulo"
            )
            df.loc[fuera_rango, "release_year"] = None

    return df


def validar_duplicados(df: pd.DataFrame, resultado: ResultadoValidacion) -> pd.DataFrame:
    """Detecta y elimina duplicados exactos por aa_id."""
    if "aa_id" not in df.columns:
        return df
    duplicados = df.duplicated(subset=["aa_id"]).sum()
    if duplicados > 0:
        resultado.advertencias.append(f"{duplicados} registros duplicados por aa_id eliminados")
        df = df.drop_duplicates(subset=["aa_id"], keep="first")
    return df


def validar_y_limpiar_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, ResultadoValidacion]:
    """
    Punto de entrada principal de la validación de esquema.
    Aplica todas las reglas y retorna el DataFrame limpio + el reporte.
    """
    resultado = ResultadoValidacion(es_valido=True, filas_originales=len(df))
    logger.info(f"Iniciando validación de esquema sobre {len(df)} filas")

    if not validar_columnas_presentes(df, resultado):
        resultado.es_valido = False
        logger.error("Validación detenida: faltan columnas críticas")
        return df, resultado

    df = validar_tipos_numericos(df, resultado)
    df = validar_pricing_tier(df, resultado)
    df = validar_rangos(df, resultado)
    df = validar_duplicados(df, resultado)

    resultado.filas_descartadas = resultado.filas_originales - len(df)
    logger.info(resultado.resumen())
    return df, resultado


if __name__ == "__main__":
    import sys
    ruta = sys.argv[1] if len(sys.argv) > 1 else "../datos/llm_price_performance_tracker_2026-03-31.csv"
    df = pd.read_csv(ruta)
    df_limpio, reporte = validar_y_limpiar_dataset(df)
    print(reporte.resumen())
