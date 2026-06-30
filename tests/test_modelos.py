"""
test_modelos.py - Pruebas de los endpoints de consulta de modelos.

Cubren:
  - GET /modelos                (listado con filtros y paginación)
  - GET /modelos/{model_slug}   (detalle por slug, incluido el caso 404)
"""


def test_listar_modelos_estructura(client):
    """GET /modelos debe devolver total, skip, limit y data."""
    resp = client.get("/modelos")
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "data" in data
    assert isinstance(data["data"], list)


def test_listar_modelos_filtra_por_tier(client):
    """El filtro por tier debe devolver solo modelos de ese tier."""
    resp = client.get("/modelos", params={"tier": "Premium"})
    assert resp.status_code == 200
    data = resp.json()
    assert all(m["pricing_tier"] == "Premium" for m in data["data"])


def test_listar_modelos_filtra_por_open_source(client):
    """El filtro open_source=true debe devolver solo modelos abiertos."""
    resp = client.get("/modelos", params={"open_source": "true"})
    assert resp.status_code == 200
    data = resp.json()
    assert all(m["is_open_source"] is True for m in data["data"])


def test_listar_modelos_limit_invalido(client):
    """Un limit fuera de rango (>1000) debe rechazarse con 422."""
    resp = client.get("/modelos", params={"limit": 5000})
    assert resp.status_code == 422


def test_obtener_modelo_existente(client):
    """GET /modelos/{slug} debe devolver el modelo correcto."""
    resp = client.get("/modelos/gpt-test-1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_slug"] == "gpt-test-1"
    assert data["provider"] == "OpenAI"


def test_obtener_modelo_inexistente_devuelve_404(client):
    """Un slug que no existe debe devolver 404 con mensaje de error."""
    resp = client.get("/modelos/no-existe-xyz")
    assert resp.status_code == 404
    assert "no encontrado" in resp.json()["detail"].lower()