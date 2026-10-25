import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    app.state.model = None
    yield TestClient(app)
    app.state.model = None


def test_health_reports_no_model_before_one_is_loaded(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": False}


def test_health_reports_a_loaded_model(client):
    app.state.model = object()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_health_only_accepts_get(client):
    assert client.post("/health").status_code == 405
