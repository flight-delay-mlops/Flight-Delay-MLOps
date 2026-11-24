"""Inference API: classifies flights by arrival delay severity."""
import logging
import os
from contextlib import asynccontextmanager

import mlflow.pyfunc
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from mlflow.exceptions import MlflowException

from api import features, live
from api.errors import (
    InputRejectedError,
    ModelNotLoadedError,
    handle_input_rejected,
    handle_model_not_loaded,
    handle_unexpected_error,
    handle_validation_error,
)
from api.schemas import ClassifyRequest, ClassifyResponse, LabelsRequest

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # so the live scores show up in `docker logs`

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


# HEAD too: some uptime checkers only ask for the headers
@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    """Report that the service is up and whether a model is loaded."""
    return {"status": "healthy", "model_loaded": app.state.model is not None}


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    """Classify each flight as on_time, moderate_delay or severe_delay, in the order received."""
    model = app.state.model
    if model is None:
        raise ModelNotLoadedError()
    flights = request.to_dataframe()
    try:
        # The latest delay rates of each flight's airport, airline and route (Feast online store)
        predictions = model.predict(features.add_online_features(flights))
    except MlflowException as error:
        if error.error_code != "INVALID_PARAMETER_VALUE":
            raise
        # MLflow starts its message with a copy of the data and the schema: only the reason helps
        raise InputRejectedError(str(error).rsplit("Error: ", 1)[-1]) from error
    classifications = [str(label) for label in predictions]
    # Kept until the true labels arrive on /labels (only the leaderboard sends a request_id)
    # A storage problem (e.g. a full disk) must not cost us the answer, which is what gets scored
    if request.request_id:
        try:
            live.save_request(request.request_id, request.mode, flights, classifications)
        except Exception:
            logger.exception("Could not save request %s", request.request_id)
    return ClassifyResponse(classifications=classifications)


@app.post("/labels")
def labels(request: LabelsRequest):
    """Receive the true labels of an earlier /classify request and score our predictions."""
    saved = live.save_labels(request.request_id, request.labels, request.outcomes)
    if saved is None:
        logger.warning("Labels for unknown (or already labelled) request %s", request.request_id)
    else:
        flights, scores = saved
        # Real outcomes refresh the feature store. Only live ones: the sandbox is January
        # flights, which would mix old outcomes into the current rates
        if request.mode == "live":
            features.learn_outcomes(flights, request.outcomes)
        # mode tells live traffic (scored) from the January sandbox, so they never get mixed up
        logger.info("%s request %s: accuracy %.3f, macro-F1 %.3f", request.mode,
                       request.request_id, scores["accuracy"], scores["macro_f1"])
    # The leaderboard only needs a 200; anything else counts as a failed delivery
    return {"status": "received"}
