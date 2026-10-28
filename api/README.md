# Inference API

FastAPI service that classifies flights by arrival delay severity, with the model
registered in MLflow. This is the endpoint the centralized evaluation calls.

| File | What it holds |
|---|---|
| [main.py](main.py) | The app: loads the model at startup, `GET /health`, `POST /classify` |
| [schemas.py](schemas.py) | Request and response shapes (the 14 BTS columns of a flight) |
| [errors.py](errors.py) | Error responses (422, 503, 500), all as `{"detail": ...}` |
| [requirements.txt](requirements.txt) | The API's packages, same versions as training |
| [Dockerfile](Dockerfile) | The image that runs the API |
| [Dockerfile.dockerignore](Dockerfile.dockerignore) | Which files the image build receives |

## Endpoints

**`GET /health`**: is the service up, and is a model loaded?

```json
{"status": "healthy", "model_loaded": true}
```

**`POST /classify`**: one label per flight, in the order received.

```json
{"instances": [{"FL_DATE": "2025-05-15", "OP_UNIQUE_CARRIER": "9E", "ORIGIN": "DTW", "...": "..."}]}
```

```json
{"classifications": ["on_time"]}
```

Flights use the raw BTS column names. Unknown columns are ignored (the evaluation
may send complete BTS rows) and missing ones count as null.

| Status | When |
|---|---|
| 200 | Classified |
| 422 | Malformed request (says which instance and field), or the model rejected the input (says why) |
| 503 | The app is up but no model is loaded (e.g. MLflow was unreachable at startup) |
| 500 | Anything unexpected; no internal details are returned, the traceback is in the server log |

FastAPI also serves interactive docs at `/docs` while the app runs.

## Configuration

Environment variables, read at startup:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_URI` | `models:/flight-delay@production` | Model to serve: the version the `@production` alias points to |
| `MLFLOW_TRACKING_URI` | – | MLflow server, `http://10.17.0.167:5000` |
| `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | – | MLflow login |

The model is loaded **once, at startup**. If loading fails the app still starts and
answers 503, so `/health` shows what's wrong. After moving `@production` to another
version, restart the API to serve it.

## Run it locally

From the repository root, in the project's virtual environment:

```bash
pip install -r requirements-dev.txt
uvicorn api.main:app --reload
```

Then open http://127.0.0.1:8000/docs. Set the three `MLFLOW_*` variables first
(see `.env.template`) to serve the real model; without them it runs with no
model and `/classify` answers 503.

## Run it with Docker

Build from the **repository root** (the code is imported as the `api` package):

```bash
docker build -f api/Dockerfile -t flight-delay-api .
docker run -d --name flight-delay-api -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://10.17.0.167:5000 \
  -e MLFLOW_TRACKING_USERNAME=... -e MLFLOW_TRACKING_PASSWORD=... \
  flight-delay-api
```

`docker ps` shows the container as `healthy` once `/health` answers (Docker
checks it every 30 s). `docker logs flight-delay-api` shows the server log.

Choices in the [Dockerfile](Dockerfile):

- **Python 3.12**, like the training image: the model is pickled there, and a
  scikit-learn pickle may not load under another Python version.
- **Requirements installed before the code is copied**: Docker caches that slow
  layer, so changing the code rebuilds in seconds.
- **Only `api/` is sent to the build**: `Dockerfile.dockerignore` (picked up
  automatically because it's named after the Dockerfile) replaces the root
  `.dockerignore`, which only lets the training image see `requirements.txt`.
  This needs BuildKit, Docker's default builder since version 23.
- **Runs as a normal user**, not root.
- **`HEALTHCHECK`** so Docker itself knows when the app stops answering.

## Published images

CI builds this image for every PR and push, and on `main` publishes it to GitHub's
container registry as:

```
ghcr.io/flight-delay-mlops/flight-delay-mlops/api:<commit SHA>
```

One tag per commit, never moved, so any version can be deployed or rolled back
to by its SHA. See [.github/workflows/README.md](../.github/workflows/README.md).
