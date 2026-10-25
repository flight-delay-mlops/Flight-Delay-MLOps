"""Inference API: classifies flights by arrival delay severity."""
from fastapi import FastAPI

app = FastAPI(title="Flight delay inference API")
app.state.model = None


@app.get("/health")
def health():
    """Report that the service is up and whether a model is loaded."""
    return {"status": "healthy", "model_loaded": app.state.model is not None}
