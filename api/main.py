"""
main.py - API REST para el proyecto LLM Price Performance
Endpoints para consultar modelos, predicciones y estadísticas desde MongoDB
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import pickle
import os
from api.database import conectar_mongo, cargar_dataset_a_mongo, obtener_coleccion

# ── Cargar modelos ML al iniciar ──────────────────────────────────────────────
MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "trained_models")

def cargar_modelos():
    modelos = {}
    try:
        with open(os.path.join(MODELS_PATH, "rf_optimizado.pkl"), "rb") as f:
            modelos["rf"] = pickle.load(f)
        with open(os.path.join(MODELS_PATH, "gb_optimizado.pkl"), "rb") as f:
            modelos["gb"] = pickle.load(f)
        with open(os.path.join(MODELS_PATH, "scaler.pkl"), "rb") as f:
            modelos["scaler"] = pickle.load(f)
        print("✓ Modelos ML cargados correctamente")
    except Exception as e:
        print(f"⚠ No se pudieron cargar los modelos: {e}")
    return modelos

modelos_ml = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global modelos_ml
    print(" Iniciando API...")
    conectar_mongo()
    cargar_dataset_a_mongo()
    modelos_ml = cargar_modelos()
    yield
    # Shutdown
    print(" Cerrando API...")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LLM Price Performance API",
    description="API para consultar modelos LLM, predicciones de pricing y análisis de valor",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def doc_to_dict(doc):
    """Convierte documento MongoDB a dict serializable."""
    doc.pop("_id", None)
    for k, v in doc.items():
        if isinstance(v, float) and np.isnan(v):
            doc[k] = None
    return doc

FEATURES_CLASIFICACION = [
    "aa_intelligence_index", "aa_coding_index", "aa_math_index",
    "input_cost_usd_per_1m", "output_cost_usd_per_1m",
    "output_tokens_per_second", "time_to_first_token_s",
    "chatbot_arena_elo", "release_year", "cost_avg"
]

# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "mensaje": "API LLM Price Performance funcionando",
        "endpoints": ["/modelos", "/modelos/{nombre}", "/stats", "/prediccion", "/top-valor", "/providers"]
    }

@app.get("/health", tags=["Health"])
def health():
    col = obtener_coleccion()
    total = col.count_documents({})
    return {"status": "ok", "modelos_en_db": total, "modelos_ml_cargados": list(modelos_ml.keys())}

# ── Modelos ───────────────────────────────────────────────────────────────────

@app.get("/modelos", tags=["Modelos"])
def listar_modelos(
    provider: str = Query(None, description="Filtrar por proveedor"),
    tier: str = Query(None, description="Filtrar por pricing_tier"),
    open_source: bool = Query(None, description="Filtrar por open source"),
    limit: int = Query(20, ge=1, le=1000, description="Límite de resultados"),
    skip: int = Query(0, ge=0, description="Offset para paginación")
):
    """Lista modelos con filtros opcionales y paginación."""
    col = obtener_coleccion()
    filtro = {}
    if provider:
        filtro["provider"] = {"$regex": provider, "$options": "i"}
    if tier:
        filtro["pricing_tier"] = tier
    if open_source is not None:
        filtro["is_open_source"] = open_source

    total = col.count_documents(filtro)
    docs  = list(col.find(filtro, {"_id": 0}).skip(skip).limit(limit))
    return {"total": total, "skip": skip, "limit": limit, "data": [doc_to_dict(d) for d in docs]}

@app.get("/modelos/{model_slug}", tags=["Modelos"])
def obtener_modelo(model_slug: str):
    """Obtiene un modelo específico por su slug."""
    col = obtener_coleccion()
    doc = col.find_one({"model_slug": model_slug}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Modelo '{model_slug}' no encontrado")
    return doc_to_dict(doc)

# ── Estadísticas ──────────────────────────────────────────────────────────────

@app.get("/stats", tags=["Estadísticas"])
def estadisticas_generales():
    """Estadísticas globales del dataset."""
    col = obtener_coleccion()
    total = col.count_documents({})

    tiers = col.aggregate([
        {"$group": {"_id": "$pricing_tier", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ])
    providers = col.aggregate([
        {"$group": {"_id": "$provider", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ])
    open_src = col.aggregate([
        {"$group": {"_id": "$is_open_source", "count": {"$sum": 1}}}
    ])
    return {
        "total_modelos": total,
        "por_tier": {d["_id"]: d["count"] for d in tiers},
        "top_10_providers": {d["_id"]: d["count"] for d in providers},
        "open_source": {str(d["_id"]): d["count"] for d in open_src}
    }

@app.get("/stats/benchmarks", tags=["Estadísticas"])
def estadisticas_benchmarks():
    """Estadísticas de benchmarks agrupadas por pricing_tier."""
    col = obtener_coleccion()
    pipeline = [
        {"$match": {"aa_intelligence_index": {"$ne": None}, "pricing_tier": {"$ne": "Unknown"}}},
        {"$group": {
            "_id": "$pricing_tier",
            "avg_intelligence": {"$avg": "$aa_intelligence_index"},
            "avg_coding":       {"$avg": "$aa_coding_index"},
            "avg_math":         {"$avg": "$aa_math_index"},
            "avg_speed":        {"$avg": "$output_tokens_per_second"},
            "count":            {"$sum": 1}
        }},
        {"$sort": {"avg_intelligence": -1}}
    ]
    resultado = list(col.aggregate(pipeline))
    for r in resultado:
        r.pop("_id_mongo", None)
        r["tier"] = r.pop("_id")
        for k, v in r.items():
            if isinstance(v, float) and np.isnan(v):
                r[k] = None
    return resultado

@app.get("/providers", tags=["Estadísticas"])
def listar_providers():
    """Lista todos los providers con estadísticas."""
    col = obtener_coleccion()
    pipeline = [
        {"$group": {
            "_id": "$provider",
            "total_modelos":      {"$sum": 1},
            "avg_intelligence":   {"$avg": "$aa_intelligence_index"},
            "avg_costo_blended":  {"$avg": "$blended_cost_usd_per_1m"},
            "modelos_open_source":{"$sum": {"$cond": ["$is_open_source", 1, 0]}}
        }},
        {"$sort": {"total_modelos": -1}}
    ]
    resultado = list(col.aggregate(pipeline))
    for r in resultado:
        r["provider"] = r.pop("_id")
        for k, v in r.items():
            if isinstance(v, float) and (np.isnan(v) if isinstance(v, float) else False):
                r[k] = None
    return resultado

# ── Top valor ─────────────────────────────────────────────────────────────────

@app.get("/top-valor", tags=["Análisis"])
def top_modelos_por_valor(
    n: int = Query(10, ge=1, le=50),
    tier: str = Query(None)
):
    """Top N modelos con mejor intelligence_per_dollar (los más baratos por unidad de inteligencia)."""
    col = obtener_coleccion()
    filtro = {"intelligence_per_dollar": {"$gt": 0}}
    if tier:
        filtro["pricing_tier"] = tier
    docs = list(
        col.find(filtro, {"_id": 0})
        .sort("intelligence_per_dollar", -1)
        .limit(n)
    )
    return [doc_to_dict(d) for d in docs]

@app.get("/top-inteligencia", tags=["Análisis"])
def top_modelos_inteligencia(n: int = Query(10, ge=1, le=50)):
    """Top N modelos por aa_intelligence_index."""
    col = obtener_coleccion()
    docs = list(
        col.find({"aa_intelligence_index": {"$gt": 0}}, {"_id": 0})
        .sort("aa_intelligence_index", -1)
        .limit(n)
    )
    return [doc_to_dict(d) for d in docs]

# ── Predicción ML ─────────────────────────────────────────────────────────────

@app.post("/prediccion", tags=["Machine Learning"])
def predecir_tier(datos: dict):
    """
    Predice el pricing_tier de un modelo basado en sus características.
    
    Ejemplo de body:
    {
        "aa_intelligence_index": 45.0,
        "aa_coding_index": 40.0,
        "aa_math_index": 38.0,
        "input_cost_usd_per_1m": 3.0,
        "output_cost_usd_per_1m": 12.0,
        "output_tokens_per_second": 80.0,
        "time_to_first_token_s": 1.5,
        "chatbot_arena_elo": 1200.0,
        "release_year": 2025.0
    }
    """
    if not modelos_ml:
        raise HTTPException(status_code=503, detail="Modelos ML no disponibles")

    try:
        datos["cost_avg"] = (
            float(datos.get("input_cost_usd_per_1m", 0) or 0) +
            float(datos.get("output_cost_usd_per_1m", 0) or 0)
        ) / 2
        valores = [float(datos.get(f, 0) or 0) for f in FEATURES_CLASIFICACION]
        X = np.array(valores).reshape(1, -1)
        X_scaled = modelos_ml["scaler"].transform(X)

        pred_rf = modelos_ml["rf"].predict(X_scaled)[0]
        pred_gb = modelos_ml["gb"].predict(X_scaled)[0]

        proba_rf = modelos_ml["rf"].predict_proba(X_scaled)[0]
        clases   = modelos_ml["rf"].classes_.tolist()
        confianza = {c: round(float(p), 3) for c, p in zip(clases, proba_rf)}

        return {
            "prediccion_rf": pred_rf,
            "prediccion_gb": pred_gb,
            "consenso": pred_rf if pred_rf == pred_gb else f"RF={pred_rf} | GB={pred_gb}",
            "confianza_rf": confianza,
            "features_usadas": FEATURES_CLASIFICACION
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {str(e)}")

@app.get("/prediccion/tier/{tier}", tags=["Machine Learning"])
def modelos_reales_vs_predichos(tier: str):
    """Compara los modelos de un tier con lo que predice el modelo RF."""
    if not modelos_ml:
        raise HTTPException(status_code=503, detail="Modelos ML no disponibles")

    col  = obtener_coleccion()
    docs = list(col.find({"pricing_tier": tier, "aa_intelligence_index": {"$ne": None}}, {"_id": 0}).limit(50))
    resultados = []
    for doc in docs:
        try:
            valores  = [float(doc.get(f) or 0) for f in FEATURES_CLASIFICACION]
            X        = np.array(valores).reshape(1, -1)
            X_scaled = modelos_ml["scaler"].transform(X)
            pred     = modelos_ml["rf"].predict(X_scaled)[0]
            resultados.append({
                "model_name":   doc.get("model_name"),
                "provider":     doc.get("provider"),
                "tier_real":    tier,
                "tier_predicho": pred,
                "coincide":     tier == pred
            })
        except:
            continue
    return resultados
