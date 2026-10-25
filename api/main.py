"""Inference API: classifies flights by arrival delay severity."""
from fastapi import FastAPI

app = FastAPI(title="Flight delay inference API")
app.state.model = None
