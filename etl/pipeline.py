"""
etl/pipeline.py
Pipeline ETL completo: Extract -> Transform -> Load

Fuentes de datos integradas (mínimo 3, requerido por la pauta):
  1. CSV local       -> datos/llm_price_performance_tracker_2026-03-31.csv
  2. API REST externa -> mindicador.cl (tipo de cambio USD/CLP, Banco Central de Chile)
  3. MongoDB Atlas    -> destino de carga y fuente de lectura para la API/dashboard

Flujo:
  EXTRACT  -> leer CSV + consultar API externa de tipo de cambio
  TRANSFORM-> validar esquema, limpiar datos, enriquecer con conversión CLP
  LOAD     -> insertar en MongoDB Atlas con manejo de errores e idempotencia
"""

import os
import sys
import logging
import argparse
from datetime import datetime

import pandas as pd
import numpy as np
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure, BulkWriteError
from dotenv import load_dotenv

sys.path.append(os.path.dirname(__file__))
from extract_indicador import obtener_dolar_observado, convertir_usd_a_clp
from etl.validate_schema import validar_y_limpiar_dataset

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "..", "logs", "etl_pipeline.log"))
    ]
)
logger = logging.getLogger("etl.pipeline")

# ── Configuración ──────────────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "llm_pricing")
MONGO_COL = os.getenv("MONGO_COL", "modelos")
CSV_PATH = os.getenv(
    "CSV_PATH",
    os.path.join(os.path.dirname(__file__), "..", "datos", "llm_price_performance_tracker_2026-03-31.csv")
)


# ── EXTRACT ──────────────────────────────────────────────────────────────────
def extract_csv(ruta: str) -> pd.DataFrame:
    """Fuente 1: extrae el dataset crudo desde el archivo CSV."""
    logger.info(f"[EXTRACT] Leyendo CSV desde: {ruta}")
    try:
        df = pd.read_csv(ruta)
        logger.info(f"[EXTRACT] OK CSV leido: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    except FileNotFoundError:
        logger.error(f"[EXTRACT] ERROR Archivo no encontrado: {ruta}")
        raise
    except pd.errors.EmptyDataError:
        logger.error("[EXTRACT] ERROR El archivo CSV esta vacio")
        raise
    except Exception as e:
        logger.error(f"[EXTRACT] ERROR inesperado leyendo CSV: {e}")
        raise


def extract_tipo_cambio() -> dict:
    """Fuente 2: extrae el tipo de cambio actual desde la API REST externa."""
    logger.info("[EXTRACT] Consultando API externa de tipo de cambio (mindicador.cl)")
    indicador = obtener_dolar_observado()
    if indicador["es_fallback"]:
        logger.warning(
            f"[EXTRACT] ADVERTENCIA: se uso valor de respaldo para el dolar: ${indicador['valor']}"
        )
    else:
        logger.info(f"[EXTRACT] OK Tipo de cambio obtenido: ${indicador['valor']} CLP")
    return indicador


# ── TRANSFORM ────────────────────────────────────────────────────────────────
def transform(df: pd.DataFrame, indicador_dolar: dict):
    """
    Aplica validación de esquema, limpieza y enriquecimiento de datos.
    Enriquecimiento: agrega columnas de precio convertido a CLP usando
    el tipo de cambio obtenido de la API externa (fuente 2 integrada con fuente 1).
    """
    logger.info("[TRANSFORM] Validando esquema del dataset")
    df_limpio, reporte_validacion = validar_y_limpiar_dataset(df)

    if not reporte_validacion.es_valido:
        logger.error("[TRANSFORM] ERROR: el dataset no paso la validacion de esquema obligatoria")
        raise ValueError("Esquema de datos invalido: " + "; ".join(reporte_validacion.errores))

    logger.info("[TRANSFORM] Enriqueciendo con tipo de cambio USD/CLP")
    valor_dolar = indicador_dolar["valor"]

    df_limpio["blended_cost_clp_per_1m"] = df_limpio["blended_cost_usd_per_1m"].apply(
        lambda x: convertir_usd_a_clp(x, valor_dolar) if pd.notna(x) else None
    )
    df_limpio["input_cost_clp_per_1m"] = df_limpio["input_cost_usd_per_1m"].apply(
        lambda x: convertir_usd_a_clp(x, valor_dolar) if pd.notna(x) else None
    )
    df_limpio["output_cost_clp_per_1m"] = df_limpio["output_cost_usd_per_1m"].apply(
        lambda x: convertir_usd_a_clp(x, valor_dolar) if pd.notna(x) else None
    )
    df_limpio["tipo_cambio_usado"] = valor_dolar
    df_limpio["fecha_tipo_cambio"] = indicador_dolar["fecha"]
    df_limpio["fuente_tipo_cambio"] = indicador_dolar["fuente"]
    df_limpio["etl_processed_at"] = datetime.utcnow().isoformat()

    logger.info(f"[TRANSFORM] OK Dataset transformado: {len(df_limpio)} filas finales")
    return df_limpio, {
        "validacion": reporte_validacion.resumen(),
        "filas_finales": len(df_limpio),
        "tipo_cambio": valor_dolar,
    }


# ── LOAD ─────────────────────────────────────────────────────────────────────
def limpiar_nans(doc: dict) -> dict:
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in doc.items()}


def load_mongo(df: pd.DataFrame, forzar_recarga: bool = False) -> int:
    """Fuente 3: carga el dataset transformado a MongoDB Atlas con manejo de errores."""
    logger.info(f"[LOAD] Conectando a MongoDB: {MONGO_DB}.{MONGO_COL}")
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
        client.admin.command("ping")
        col = client[MONGO_DB][MONGO_COL]
    except ConnectionFailure as e:
        logger.error(f"[LOAD] ERROR: no se pudo conectar a MongoDB: {e}")
        raise

    total_existente = col.count_documents({})
    if total_existente > 0 and not forzar_recarga:
        logger.info(f"[LOAD] Coleccion ya tiene {total_existente} documentos. Use --forzar para recargar.")
        return total_existente

    if forzar_recarga and total_existente > 0:
        col.drop()
        logger.info("[LOAD] Coleccion anterior eliminada (recarga forzada)")

    documentos = [limpiar_nans(d) for d in df.to_dict(orient="records")]

    insertados = 0
    errores = 0
    BATCH = 500
    for i in range(0, len(documentos), BATCH):
        lote = documentos[i:i + BATCH]
        try:
            col.insert_many(lote, ordered=False)
            insertados += len(lote)
        except BulkWriteError as bwe:
            exitosos = bwe.details.get("nInserted", 0)
            fallidos = len(lote) - exitosos
            insertados += exitosos
            errores += fallidos
            logger.warning(f"[LOAD] ADVERTENCIA lote {i//BATCH}: {fallidos} documentos fallaron (posibles duplicados)")

    try:
        col.create_index([("model_slug", ASCENDING)])
        col.create_index([("provider", ASCENDING)])
        col.create_index([("pricing_tier", ASCENDING)])
        col.create_index([("is_open_source", ASCENDING)])
    except Exception as e:
        logger.warning(f"[LOAD] ADVERTENCIA: no se pudieron crear todos los indices: {e}")

    logger.info(f"[LOAD] OK {insertados} documentos insertados, {errores} errores")
    return insertados


# ── ORQUESTADOR ──────────────────────────────────────────────────────────────
def ejecutar_pipeline(forzar_recarga: bool = False) -> dict:
    """Ejecuta el pipeline ETL completo de punta a punta con manejo de errores global."""
    inicio = datetime.utcnow()
    logger.info("=" * 70)
    logger.info("INICIANDO PIPELINE ETL - LLM Price Performance")
    logger.info("=" * 70)

    resumen = {"exito": False, "etapas": {}}

    try:
        df_raw = extract_csv(CSV_PATH)
        resumen["etapas"]["extract_csv"] = "OK"

        indicador = extract_tipo_cambio()
        resumen["etapas"]["extract_api_externa"] = (
            "OK (fallback)" if indicador["es_fallback"] else "OK (en vivo)"
        )

        df_transformado, info_transform = transform(df_raw, indicador)
        resumen["etapas"]["transform"] = "OK"
        resumen.update(info_transform)

        insertados = load_mongo(df_transformado, forzar_recarga=forzar_recarga)
        resumen["etapas"]["load_mongo"] = "OK"
        resumen["documentos_en_db"] = insertados

        resumen["exito"] = True

    except Exception as e:
        logger.error(f"PIPELINE FALLO: {e}")
        resumen["exito"] = False
        resumen["error"] = str(e)

    duracion = (datetime.utcnow() - inicio).total_seconds()
    resumen["duracion_segundos"] = round(duracion, 2)

    logger.info("=" * 70)
    logger.info(f"PIPELINE {'COMPLETADO' if resumen['exito'] else 'FALLO'} en {duracion:.2f}s")
    logger.info("=" * 70)

    return resumen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline ETL LLM Price Performance")
    parser.add_argument("--forzar", action="store_true", help="Fuerza la recarga completa de MongoDB")
    args = parser.parse_args()

    resultado = ejecutar_pipeline(forzar_recarga=args.forzar)
    print(resultado)
    sys.exit(0 if resultado["exito"] else 1)
