"""Inference API: classifies flights by arrival delay severity."""
import os

import mlflow.pyfunc
from fastapi import FastAPI

MODEL_URI = os.getenv("MODEL_URI", "models:/flight-delay@production")


def load_model():
    return mlflow.pyfunc.load_model(MODEL_URI)


app = FastAPI(title="Flight delay inference API")
app.state.model = None


@app.get("/health")
def health():
    """Report that the service is up and whether a model is loaded."""
    return {"status": "healthy", "model_loaded": app.state.model is not None}
