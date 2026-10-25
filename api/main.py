"""Inference API: classifies flights by arrival delay severity."""
import logging
import os
from contextlib import asynccontextmanager

import mlflow.pyfunc
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from mlflow.exceptions import MlflowException

from api.errors import (
    InputRejectedError,
    ModelNotLoadedError,
    handle_input_rejected,
    handle_model_not_loaded,
    handle_unexpected_error,
    handle_validation_error,
)
from api.schemas import ClassifyRequest, ClassifyResponse

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
app.add_exception_handler(InputRejectedError, handle_input_rejected)
app.add_exception_handler(ModelNotLoadedError, handle_model_not_loaded)
app.add_exception_handler(Exception, handle_unexpected_error)
app.state.model = None


@app.get("/health")
def health():
    """Report that the service is up and whether a model is loaded."""
    return {"status": "healthy", "model_loaded": app.state.model is not None}


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    """Classify each flight as on_time, moderate_delay or severe_delay, in the order received."""
    model = app.state.model
    if model is None:
        raise ModelNotLoadedError()
    try:
        predictions = model.predict(request.to_dataframe())
    except MlflowException as error:
        if error.error_code != "INVALID_PARAMETER_VALUE":
            raise
        # MLflow starts its message with a copy of the data and the schema: only the reason helps
        raise InputRejectedError(str(error).rsplit("Error: ", 1)[-1]) from error
    return ClassifyResponse(classifications=[str(label) for label in predictions])
