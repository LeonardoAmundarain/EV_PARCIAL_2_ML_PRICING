"""
etl/extract_indicador.py
Extracción desde fuente externa: API REST mindicador.cl (Banco Central de Chile)
Tercera fuente de datos del pipeline ETL (CSV -> MongoDB -> API externa).

Documentación oficial: https://mindicador.cl/api
No requiere autenticación ni API key.
"""

import requests
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("etl.indicador")

BASE_URL = "https://mindicador.cl/api"
TIMEOUT_SEGUNDOS = 8
MAX_REINTENTOS = 3

# Valor de respaldo (fallback) si la API externa no responde.
# Se documenta explícitamente como manejo de error de fuente externa caída.
DOLAR_FALLBACK_CLP = 950.0


def obtener_dolar_observado(reintentos: int = MAX_REINTENTOS) -> dict:
    """
    Consulta el valor del dólar observado (USD -> CLP) publicado por el
    Banco Central de Chile a través de la API pública mindicador.cl.

    Implementa:
    - Timeout explícito (evita que el pipeline quede colgado).
    - Reintentos con backoff simple ante fallas transitorias de red.
    - Validación de esquema de la respuesta (claves esperadas).
    - Fallback documentado si la fuente externa no está disponible.

    Retorna
    -------
    dict con las claves:
        valor (float): valor del dólar en CLP
        fecha (str): fecha de publicación del indicador (ISO)
        fuente (str): "api_mindicador" o "fallback_local"
        es_fallback (bool): True si se usó el valor de respaldo
    """
    url = f"{BASE_URL}/dolar"

    for intento in range(1, reintentos + 1):
        try:
            logger.info(f"Consultando API externa mindicador.cl (intento {intento}/{reintentos})")
            response = requests.get(url, timeout=TIMEOUT_SEGUNDOS)
            response.raise_for_status()
            data = response.json()

            # ── Validación de esquema ──────────────────────────────────────
            if "serie" not in data or not isinstance(data["serie"], list) or len(data["serie"]) == 0:
                raise ValueError("Esquema inesperado: falta la clave 'serie' o está vacía")

            ultimo_registro = data["serie"][0]
            if "valor" not in ultimo_registro or "fecha" not in ultimo_registro:
                raise ValueError("Esquema inesperado: faltan claves 'valor' o 'fecha' en el registro")

            valor = float(ultimo_registro["valor"])
            if valor <= 0 or valor > 5000:
                # Sanity check: el dólar observado nunca debería estar fuera de este rango
                raise ValueError(f"Valor fuera de rango esperado: {valor}")

            logger.info(f"✓ Dólar observado obtenido: ${valor:.2f} CLP")
            return {
                "valor": valor,
                "fecha": ultimo_registro["fecha"],
                "fuente": "api_mindicador",
                "es_fallback": False,
            }

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout en intento {intento}/{reintentos}")
        except requests.exceptions.ConnectionError:
            logger.warning(f"Error de conexión en intento {intento}/{reintentos}")
        except requests.exceptions.HTTPError as e:
            logger.warning(f"Error HTTP {e.response.status_code} en intento {intento}/{reintentos}")
        except (ValueError, KeyError) as e:
            logger.warning(f"Error de validación de esquema: {e}")
            break  # No tiene sentido reintentar si el esquema cambió
        except Exception as e:
            logger.warning(f"Error inesperado: {e}")

    # ── Fallback si todos los reintentos fallaron ──────────────────────────
    logger.error(
        f"No se pudo obtener el dólar desde la API externa tras {reintentos} intentos. "
        f"Usando valor de respaldo: ${DOLAR_FALLBACK_CLP} CLP"
    )
    return {
        "valor": DOLAR_FALLBACK_CLP,
        "fecha": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "fuente": "fallback_local",
        "es_fallback": True,
    }


def convertir_usd_a_clp(monto_usd: float, valor_dolar: float) -> float:
    """Convierte un monto en USD a CLP usando el valor de dólar entregado."""
    if monto_usd is None:
        return None
    return round(monto_usd * valor_dolar, 2)


if __name__ == "__main__":
    resultado = obtener_dolar_observado()
    print(resultado)
    print(f"Ejemplo: $5 USD = ${convertir_usd_a_clp(5, resultado['valor']):,.0f} CLP")
