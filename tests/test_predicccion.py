"""
test_prediccion.py - Pruebas del endpoint de Machine Learning POST /prediccion.

Cubren:
  - Predicción exitosa con un payload válido.
  - Forma de la respuesta (prediccion_rf, prediccion_gb, consenso, confianza_rf).
  - Manejo de error cuando los modelos ML no están disponibles (503).
"""

import api.main as main


PAYLOAD_VALIDO = {
    "aa_intelligence_index": 45.0,
    "aa_coding_index": 40.0,
    "aa_math_index": 38.0,
    "input_cost_usd_per_1m": 3.0,
    "output_cost_usd_per_1m": 12.0,
    "output_tokens_per_second": 80.0,
    "time_to_first_token_s": 1.5,
    "chatbot_arena_elo": 1200.0,
    "release_year": 2025.0,
}


def test_prediccion_exitosa(client):
    """Un payload válido debe devolver 200 y una predicción."""
    resp = client.post("/prediccion", json=PAYLOAD_VALIDO)
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediccion_rf"] == "Premium"  # valor fijo del modelo falso
    assert data["prediccion_gb"] == "Premium"


def test_prediccion_incluye_confianza(client):
    """La respuesta debe incluir un diccionario de confianza por tier."""
    data = client.post("/prediccion", json=PAYLOAD_VALIDO).json()
    assert "confianza_rf" in data
    assert isinstance(data["confianza_rf"], dict)
    # Las probabilidades deben sumar aproximadamente 1
    assert abs(sum(data["confianza_rf"].values()) - 1.0) < 0.01


def test_prediccion_consenso(client):
    """Si RF y GB coinciden, el consenso debe ser ese tier."""
    data = client.post("/prediccion", json=PAYLOAD_VALIDO).json()
    assert data["consenso"] == "Premium"


def test_prediccion_payload_vacio_usa_defaults(client):
    """Un payload vacío no debe romper: el endpoint usa 0 por defecto."""
    resp = client.post("/prediccion", json={})
    assert resp.status_code == 200


def test_prediccion_sin_modelos_devuelve_503(client, monkeypatch):
    """Si los modelos ML no están cargados, debe responder 503."""
    monkeypatch.setattr(main, "modelos_ml", {})
    resp = client.post("/prediccion", json=PAYLOAD_VALIDO)
    assert resp.status_code == 503
    assert "no disponibles" in resp.json()["detail"].lower()