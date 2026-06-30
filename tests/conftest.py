"""
conftest.py - Configuración compartida de pytest para los tests de la API.

Estrategia de testing
----------------------
La API depende de dos servicios externos en producción:
  1. MongoDB (vía api.database)
  2. Modelos ML serializados (.pkl en models/trained_models)

Para que las pruebas unitarias sean reproducibles y NO requieran levantar
MongoDB ni cargar los .pkl reales, aquí se construyen "dobles de prueba"
(mocks/fakes) que imitan el comportamiento mínimo necesario:

  - FakeColeccion: imita los métodos de una colección de PyMongo que usa la API
    (count_documents, find, find_one, aggregate).
  - FakeCursor: imita el cursor encadenable de PyMongo (.skip().limit().sort()).
  - modelos_ml: se parchea con un RandomForest/GradientBoosting "falsos" que
    devuelven una predicción fija, evitando depender del scaler/modelos reales.

Esto permite probar la LÓGICA de los endpoints (códigos de estado, forma de la
respuesta, manejo de errores) de forma aislada y rápida.
"""

import sys
import os
import numpy as np
import pytest

# Asegura que la raíz del proyecto esté en el path para importar `api`
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)


# ──────────────────────────────────────────────────────────────────────────────
# Datos de ejemplo (simulan documentos de MongoDB)
# ──────────────────────────────────────────────────────────────────────────────
DOCS_EJEMPLO = [
    {
        "model_slug": "gpt-test-1",
        "model_name": "GPT Test 1",
        "provider": "OpenAI",
        "pricing_tier": "Premium",
        "is_open_source": False,
        "aa_intelligence_index": 50.0,
        "aa_coding_index": 48.0,
        "aa_math_index": 45.0,
        "blended_cost_usd_per_1m": 10.0,
        "intelligence_per_dollar": 5.0,
        "output_tokens_per_second": 90.0,
    },
    {
        "model_slug": "llama-test-2",
        "model_name": "Llama Test 2",
        "provider": "Meta",
        "pricing_tier": "Budget",
        "is_open_source": True,
        "aa_intelligence_index": 30.0,
        "aa_coding_index": 28.0,
        "aa_math_index": 25.0,
        "blended_cost_usd_per_1m": 1.0,
        "intelligence_per_dollar": 30.0,
        "output_tokens_per_second": 120.0,
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Dobles de prueba para MongoDB
# ──────────────────────────────────────────────────────────────────────────────
class FakeCursor:
    """Imita un cursor de PyMongo con métodos encadenables."""

    def __init__(self, docs):
        self._docs = [dict(d) for d in docs]

    def skip(self, n):
        self._docs = self._docs[n:]
        return self

    def limit(self, n):
        self._docs = self._docs[:n]
        return self

    def sort(self, campo, direccion=1):
        self._docs.sort(
            key=lambda d: (d.get(campo) is None, d.get(campo)),
            reverse=(direccion == -1),
        )
        return self

    def __iter__(self):
        return iter(self._docs)

    def __list__(self):
        return list(self._docs)


class FakeColeccion:
    """Imita una colección de PyMongo (solo lo que la API usa)."""

    def __init__(self, docs):
        self._docs = [dict(d) for d in docs]

    def count_documents(self, filtro):
        if not filtro:
            return len(self._docs)
        return len([d for d in self._docs if self._coincide(d, filtro)])

    def find(self, filtro=None, proyeccion=None):
        filtro = filtro or {}
        encontrados = [d for d in self._docs if self._coincide(d, filtro)]
        return FakeCursor(encontrados)

    def find_one(self, filtro, proyeccion=None):
        for d in self._docs:
            if self._coincide(d, filtro):
                return dict(d)
        return None

    def aggregate(self, pipeline):
        """
        Implementación mínima de aggregate. Detecta el campo de agrupación del
        primer $group con forma '$campo' y devuelve documentos
        {"_id": valor, "count": n, ...promedios en None}. Suficiente para que
        los endpoints /stats, /stats/benchmarks y /providers no rompan.
        """
        group = next((e["$group"] for e in pipeline if "$group" in e), None)
        if not group:
            return iter([dict(d) for d in self._docs])

        campo = group["_id"]
        # _id suele ser "$pricing_tier", "$provider", "$is_open_source"
        if isinstance(campo, str) and campo.startswith("$"):
            campo = campo[1:]
        else:
            campo = None

        grupos = {}
        for d in self._docs:
            clave = d.get(campo) if campo else None
            grupos.setdefault(clave, []).append(d)

        salida = []
        for clave, items in grupos.items():
            doc = {"_id": clave, "count": len(items), "total_modelos": len(items)}
            # Rellena cualquier acumulador de promedio/suma con un valor neutro
            for acc in group:
                if acc == "_id":
                    continue
                if acc not in doc:
                    doc[acc] = 0
            salida.append(doc)
        return iter(salida)

    @staticmethod
    def _coincide(doc, filtro):
        """Evalúa filtros simples de igualdad y algunos operadores ($gt, $ne, $regex)."""
        for clave, condicion in filtro.items():
            valor = doc.get(clave)
            if isinstance(condicion, dict):
                for op, ref in condicion.items():
                    if op == "$gt" and not (valor is not None and valor > ref):
                        return False
                    if op == "$ne" and valor == ref:
                        return False
                    if op == "$regex":
                        if valor is None or ref.lower() not in str(valor).lower():
                            return False
            else:
                if valor != condicion:
                    return False
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Doble de prueba para los modelos ML
# ──────────────────────────────────────────────────────────────────────────────
class FakeModeloML:
    """Modelo ML falso que devuelve una predicción fija y probabilidades."""

    classes_ = np.array(["Budget", "Mid", "Premium", "Ultra"])

    def predict(self, X):
        return np.array(["Premium"] * len(X))

    def predict_proba(self, X):
        # Probabilidades fijas para las 4 clases
        return np.array([[0.1, 0.2, 0.6, 0.1]] * len(X))


class FakeScaler:
    """Scaler falso: devuelve la entrada sin transformar."""

    def transform(self, X):
        return np.asarray(X)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures de pytest
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def fake_coleccion():
    """Devuelve una colección falsa cargada con los documentos de ejemplo."""
    return FakeColeccion(DOCS_EJEMPLO)


@pytest.fixture
def client(monkeypatch, fake_coleccion):
    """
    Cliente de pruebas de FastAPI con MongoDB y modelos ML mockeados.

    Se parchea:
      - api.database.conectar_mongo / cargar_dataset_a_mongo  -> no-op
      - api.main.obtener_coleccion                            -> colección falsa
      - api.main.modelos_ml                                   -> modelos falsos
    """
    import api.database as database
    import api.main as main

    modelos_falsos = {"rf": FakeModeloML(), "gb": FakeModeloML(), "scaler": FakeScaler()}

    # Evitar conexión real a Mongo durante el lifespan de arranque
    monkeypatch.setattr(database, "conectar_mongo", lambda: None)
    monkeypatch.setattr(database, "cargar_dataset_a_mongo", lambda *a, **k: 0)
    monkeypatch.setattr(main, "conectar_mongo", lambda: None)
    monkeypatch.setattr(main, "cargar_dataset_a_mongo", lambda *a, **k: 0)

    # Evitar que el lifespan cargue los .pkl reales: devuelve los modelos falsos
    monkeypatch.setattr(main, "cargar_modelos", lambda: modelos_falsos)

    # La colección que usan los endpoints devuelve nuestra colección falsa
    monkeypatch.setattr(main, "obtener_coleccion", lambda: fake_coleccion)

    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:
        # Tras el arranque (lifespan), forzamos los modelos falsos por si acaso
        monkeypatch.setattr(main, "modelos_ml", modelos_falsos)
        yield c