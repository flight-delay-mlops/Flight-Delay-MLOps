# Inference API image: serves POST /classify with the model registered in MLflow.
# Build it from the repository root, since the code is imported as the `api` package:
#   docker build -f api/Dockerfile -t flight-delay-api .
# Which files Docker receives is decided by api/Dockerfile.dockerignore, not the root one.

# Same Python as the training image (root Dockerfile), which saves the model:
# a pickled scikit-learn model may not load under another Python version
FROM python:3.12-slim

# No .pyc files in the image, and logs reach `docker logs` straight away
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies before the code: Docker reuses this slow layer until requirements.txt changes
COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY api/ api/

# Run as a normal user, not root
RUN useradd --create-home --uid 1000 api
USER api

EXPOSE 8000

# Docker marks the container "unhealthy" when /health stops answering
# (the slim image has no curl, so Python makes the request)
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

# Configured at `docker run` time with -e:
#   MODEL_URI                    model to serve (default models:/flight-delay@production)
#   MLFLOW_TRACKING_URI          MLflow server to load it from
#   MLFLOW_TRACKING_USERNAME/PASSWORD  MLflow login
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
