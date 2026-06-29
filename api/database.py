"""
database.py - Conexión y operaciones con MongoDB
Carga el dataset CSV a MongoDB y expone la colección para consultas
"""

import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import numpy as np
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure

# ── Config desde variables de entorno ────────────────────────────────────────
MONGO_URI    = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB     = os.getenv("MONGO_DB", "llm_pricing")
MONGO_COL    = os.getenv("MONGO_COL", "modelos")
CSV_PATH     = os.getenv("CSV_PATH", os.path.join(
    os.path.dirname(__file__), "..", "datos",
    "llm_price_performance_tracker_2026-03-31.csv"
))

_client: MongoClient = None
_coleccion = None

# ── Conexión ──────────────────────────────────────────────────────────────────
def conectar_mongo():
    global _client, _coleccion
    try:
        _client    = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        _client.admin.command("ping")
        _coleccion = _client[MONGO_DB][MONGO_COL]
        print(f"✓ Conectado a MongoDB: {MONGO_URI} → {MONGO_DB}.{MONGO_COL}")
    except ConnectionFailure as e:
        print(f"✗ No se pudo conectar a MongoDB: {e}")
        raise

def obtener_coleccion():
    if _coleccion is None:
        conectar_mongo()
    return _coleccion

# ── Carga de datos ────────────────────────────────────────────────────────────
def limpiar_nans(doc: dict) -> dict:
    """Reemplaza NaN/inf por None para que MongoDB los acepte."""
    limpio = {}
    for k, v in doc.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            limpio[k] = None
        else:
            limpio[k] = v
    return limpio

def cargar_dataset_a_mongo(forzar_recarga: bool = False):
    """
    Carga el CSV al MongoDB solo si la colección está vacía
    o si forzar_recarga=True.
    """
    col   = obtener_coleccion()
    total = col.count_documents({})

    if total > 0 and not forzar_recarga:
        print(f" MongoDB ya tiene {total} documentos. Se omite la carga.")
        return total

    print(f" Cargando dataset desde: {CSV_PATH}")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f" Archivo no encontrado: {CSV_PATH}")
        return 0

    # Limpiar NaN antes de insertar
    documentos = [limpiar_nans(row) for row in df.to_dict(orient="records")]

    if forzar_recarga:
        col.drop()
        print("🗑  Colección eliminada para recarga completa")

    # Inserción en lotes de 500 para eficiencia
    BATCH = 500
    insertados = 0
    for i in range(0, len(documentos), BATCH):
        lote = documentos[i:i+BATCH]
        col.insert_many(lote)
        insertados += len(lote)

    # Índices para búsquedas rápidas
    col.create_index([("model_slug", ASCENDING)])
    col.create_index([("provider",     ASCENDING)])
    col.create_index([("pricing_tier", ASCENDING)])
    col.create_index([("is_open_source", ASCENDING)])
    col.create_index([("aa_intelligence_index", ASCENDING)])
    col.create_index([("intelligence_per_dollar", ASCENDING)])

    print(f" {insertados} documentos insertados en MongoDB")
    return insertados
