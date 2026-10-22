# Training environment: the pipeline workflow runs `dvc repro` (prepare + train) in this image.
# Only the dependencies are baked in; the repo is mounted at run time, so DVC sees the
# current code, data and dvc.lock.
FROM python:3.12-slim
# git lets MLflow record which commit trained each model
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
