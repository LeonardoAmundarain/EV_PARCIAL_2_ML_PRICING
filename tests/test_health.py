"""
test_health.py - Pruebas de los endpoints de salud (/ y /health).

Verifican que la API responda correctamente y exponga la información básica
de estado que el dashboard usa para mostrar "API conectada".
"""


def test_root_responde_ok(client):
    """El endpoint raíz debe responder 200 y status 'ok'."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "endpoints" in data


def test_root_lista_endpoints_principales(client):
    """La raíz debe anunciar los endpoints principales del proyecto."""
    data = client.get("/").json()
    assert "/modelos" in data["endpoints"]
    assert "/prediccion" in data["endpoints"]


def test_health_responde_ok(client):
    """/health debe responder 200 con el conteo de modelos en BD."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "modelos_en_db" in data


def test_health_reporta_modelos_ml_cargados(client):
    """/health debe listar los modelos ML disponibles (rf, gb, scaler)."""
    data = client.get("/health").json()
    assert "rf" in data["modelos_ml_cargados"]
    assert "gb" in data["modelos_ml_cargados"]