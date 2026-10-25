"""Inference API: classifies flights by arrival delay severity."""
import logging
import os
from contextlib import asynccontextmanager

import mlflow.pyfunc
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from api.errors import handle_validation_error

logger = logging.getLogger(__name__)

MODEL_URI = os.getenv("MODEL_URI", "models:/flight-delay@production")


def load_model():
    return mlflow.pyfunc.load_model(MODEL_URI)


@asynccontextmanager
async def lifespan(app):
    try:
        app.state.model = load_model()
    except Exception:
        app.state.model = None
        logger.exception("Could not load the model from %s", MODEL_URI)
    yield


app = FastAPI(title="Flight delay inference API", lifespan=lifespan)
app.add_exception_handler(RequestValidationError, handle_validation_error)
app.state.model = None


@app.get("/health")
def health():
    """Report that the service is up and whether a model is loaded."""
    return {"status": "healthy", "model_loaded": app.state.model is not None}
