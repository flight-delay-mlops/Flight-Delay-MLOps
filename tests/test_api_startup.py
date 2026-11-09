from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from api import main
from api.main import app


@pytest.fixture(autouse=True)
def reset_model():
    app.state.model = None
    yield
    app.state.model = None


@pytest.fixture
def client():
    return TestClient(app)


def test_health_reports_no_model_before_one_is_loaded(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": False}


def test_health_reports_a_loaded_model(client):
    app.state.model = object()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_health_accepts_get_and_head_only(client):
    assert client.head("/health").status_code == 200
    assert client.post("/health").status_code == 405


def test_load_model_asks_mlflow_for_the_configured_model(monkeypatch):
    mlflow_loader = Mock(return_value="model")
    monkeypatch.setattr(main.mlflow.pyfunc, "load_model", mlflow_loader)

    assert main.load_model() == "model"
    mlflow_loader.assert_called_once_with(main.MODEL_URI)


def test_model_is_loaded_when_the_app_starts(monkeypatch):
    model = object()
    loader = Mock(return_value=model)
    monkeypatch.setattr(main, "load_model", loader)

    with TestClient(app) as started:
        assert started.get("/health").json()["model_loaded"] is True

    assert app.state.model is model
    loader.assert_called_once_with()


def test_app_starts_without_a_model_when_loading_fails(monkeypatch, caplog):
    monkeypatch.setattr(main, "load_model", Mock(side_effect=RuntimeError("MLflow unreachable")))

    with TestClient(app) as started:
        response = started.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is False
    assert "Could not load the model" in caplog.text


def test_failed_load_clears_a_previously_loaded_model(monkeypatch):
    app.state.model = object()
    monkeypatch.setattr(main, "load_model", Mock(side_effect=RuntimeError("MLflow unreachable")))

    with TestClient(app):
        pass

    assert app.state.model is None
