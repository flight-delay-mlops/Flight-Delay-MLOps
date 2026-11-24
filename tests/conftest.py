import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(autouse=True)
def live_db_in_tmp(tmp_path, monkeypatch):
    # Each test gets its own empty live.db instead of writing into the repository
    monkeypatch.setenv("DATA_DIR", str(tmp_path))


@pytest.fixture(autouse=True)
def reset_model():
    app.state.model = None
    yield
    app.state.model = None


@pytest.fixture
def client():
    # A 500 is re-raised by Starlette after it is answered; the tests want to see the answer
    return TestClient(app, raise_server_exceptions=False)
