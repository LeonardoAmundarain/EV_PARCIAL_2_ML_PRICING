"""
test_stats.py - Pruebas de los endpoints de estadísticas y análisis.

Cubren:
  - GET /stats        (estadísticas globales)
  - GET /top-valor    (ranking por intelligence_per_dollar)

Nota: /stats y /providers usan aggregate(); el doble de prueba FakeColeccion
devuelve los documentos sin agregar, por lo que aquí solo se valida el
contrato de respuesta (código 200 y forma esperada), no el cálculo exacto
de la agregación, que es responsabilidad de MongoDB.
"""


def test_stats_responde_ok(client):
    """GET /stats debe responder 200 e incluir total_modelos."""
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_modelos" in data
    assert data["total_modelos"] == 2  # dos documentos de ejemplo


def test_top_valor_responde_lista(client):
    """GET /top-valor debe devolver una lista de modelos."""
    resp = client.get("/top-valor", params={"n": 5})
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_top_valor_ordenado_descendente(client):
    """Los modelos deben venir ordenados por intelligence_per_dollar desc."""
    resp = client.get("/top-valor", params={"n": 5})
    datos = resp.json()
    valores = [m["intelligence_per_dollar"] for m in datos]
    assert valores == sorted(valores, reverse=True)


def test_top_valor_n_invalido(client):
    """Un n fuera de rango (>50) debe rechazarse con 422."""
    resp = client.get("/top-valor", params={"n": 999})
    assert resp.status_code == 422