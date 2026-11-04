# Infrastructure

Everything runs on two DEI VMs, reachable only through the DEI VPN.

| VM                | IP          | Runs                                                        |
| ----------------- | ----------- | ----------------------------------------------------------- |
| VM1 (4 CPU, 8 GB) | 10.17.0.167 | MLflow server, DVC store                                    |
| VM2 (2 CPU, 4 GB) | 10.17.0.156 | GitHub Actions runner `cpu`, the inference API (port 8000), later monitoring |

Both run Ubuntu 24.04 with Docker, set up by `infra/setup_vm.sh`. Log in with `ssh admin@<IP>`.

## MLflow

UI and tracking API at http://10.17.0.167:5000 (login required).

To log from Python, set these variables (see `.env.template`):

```bash
MLFLOW_TRACKING_URI=http://10.17.0.167:5000
MLFLOW_TRACKING_USERNAME=...
MLFLOW_TRACKING_PASSWORD=...
```

`src/train.py` reads them from `.env` (copy `.env.template`, it's git-ignored), so `dvc repro` logs to MLflow with your login. Each training run is logged to the `flight-delay` experiment and registered as a new version of the `flight-delay` model. Aliases point at versions:

- `@staging`: the candidate. `train.py` only moves it to a new version that passes the **quality gate**: a higher macro-F1 on the validation set than the `@production` model. Every version is tagged `quality_gate: passed` or `failed`
- `@production`: the model the API serves. To promote the candidate, run **Actions → Promote → Run workflow** on `main`: it moves `@production` to the `@staging` version and redeploys the API, which only loads the model at startup (see [Deployment](#deployment))

Load a model in Python with `mlflow.pyfunc.load_model("models:/flight-delay@production")`.

The server is defined in `infra/mlflow/`. Runs, users and artifacts live in the Docker volume `mlflow_mlflow-data`, so they survive restarts. To update it, copy `infra/mlflow/` to `~/mlflow` on VM1 and run `docker compose up -d --build` there.

## DVC

Git stores the code and small `.dvc` / `dvc.lock` files with data hashes. The data itself lives in `~/dvc-store` on VM1.

```bash
dvc pull      # get the data for the current commit
dvc repro     # run the pipeline in dvc.yaml (only stages whose inputs changed)
dvc push      # upload new data or pipeline outputs
```

`dvc pull` connects to VM1 over SSH, so you need the VPN and SSH access (your key on VM1, or `dvc remote modify --local vm1 ask_password true`).

### Adding new data

For example, the data and labels released after each evaluation round. Data folders are in `.gitignore`, so `git add` on data does nothing. Use DVC, then commit the small `.dvc` file it creates:

```bash
dvc add data/rounds/round_1                          # hash the data, creates data/rounds/round_1.dvc
git add data/rounds/round_1.dvc data/rounds/.gitignore
git commit -m "Add round 1 data"
dvc push                                             # upload the data to VM1
```

### After changing pipeline code or params.yaml

```bash
dvc repro                                            # re-runs the affected stages, updates dvc.lock
git add dvc.lock <changed files>
git commit -m "..."
dvc push                                             # upload the new outputs
```

Always commit `dvc.lock` together with the change that produced it.

### Trying different settings

```bash
dvc exp run -S train.model=stratified                # change a value in params.yaml and run the pipeline
dvc exp show                                         # table of every run's params and metrics
```

`dvc exp run` leaves `params.yaml` with the value you set, so set it back (or commit it) afterwards. Every run is also in MLflow.

The repo forces LF line endings (`.gitattributes`): DVC hashes files byte by byte, so Windows CRLF endings would make it re-run stages that didn't change.

## Training

Training is the `train` stage in `dvc.yaml` (`src/train.py`, the models in `src/models.py`, settings under `train:` in `params.yaml`). It logs each run to MLflow and registers the model, as `@staging` if it passes the quality gate (see [MLflow](#mlflow)).

- **Locally:** `dvc repro` with your `.env` filled in and the VPN on.
- **Automatically:** a merge to `main` that changes the pipeline runs `train.yml` on the runner. It pulls the data, runs `dvc repro` inside the training image (root `Dockerfile`: Python + `requirements.txt`) and pushes the outputs.
- **By hand:** Actions → Train → Run workflow, with **Retrain even if nothing changed** ticked to train even when DVC sees no changes.

To run the same container yourself:

```bash
docker build -t flight-delay-train .
docker run --rm --env-file .env -v "$PWD:$PWD" -w "$PWD" --user "$(id -u):$(id -g)" -e HOME=/tmp -e USER=$USER flight-delay-train dvc repro
```

## Deployment

The inference API ([api/README.md](../api/README.md)) runs on VM2 as the Docker container `flight-delay-api`:

- Endpoint: **http://10.17.0.156:8000/classify** (interactive docs at `/docs`, status at `/health`)
- Image: `ghcr.io/flight-delay-mlops/flight-delay-mlops/api:<commit SHA>`, published by CI
- Model: `models:/flight-delay@production`, loaded from MLflow at startup

It is deployed by [deploy.yml](../.github/workflows/deploy.yml):

- **Automatically**, after every merge into `main` (CI builds and publishes the image, then deploys it).
- **By hand**, from Actions → Deploy → Run workflow: leave the SHA empty to redeploy the latest `main`, or give an older commit's SHA to roll back to it.

If the new version doesn't answer correctly, the workflow puts the previous one back. On VM2:

```bash
docker ps --filter name=flight-delay-api      # running? "healthy"?
docker logs --tail 100 flight-delay-api       # server log, e.g. why the model didn't load
curl http://localhost:8000/health             # {"status":"healthy","model_loaded":true}
```

The container restarts by itself after a crash or a VM reboot (`--restart unless-stopped`).

## Runners

Self-hosted GitHub Actions runners pick up workflow jobs. GitHub's own runners can't reach the VMs.

| Runner  | Where                     | Labels | Notes                           |
| ------- | ------------------------- | ------ | ------------------------------- |
| `cpu` | VM2, `~/actions-runner` | `self-hosted`, `Linux`, `X64`, `deploy` | systemd service, starts on boot |
| `gpu` | João's computer lol      | defaults | offline when the PC is off      |

Jobs pick a runner by **label**, not by name. The extra `deploy` label exists only on VM2's runner, so the deploy job always runs there (the API must run on VM2, not on whichever machine is free). To add it: Settings → Actions → Runners → `cpu` → edit the labels → add `deploy`. Without it, deploy jobs wait in the queue with "Waiting for a runner".

Check status on GitHub: Settings → Actions → Runners.
On VM2 you can check status with:

```bash
# Go to the runner folder (the svc.sh commands below run from here)
cd ~/actions-runner

# Is the runner running? Shows its state, since when, and the last log lines
sudo ./svc.sh status

# Follow the runner's logs live (jobs picked up, finished, errors). Ctrl+C stops watching, not the runner
journalctl -u actions.runner.flight-delay-mlops-Flight-Delay-MLOps.cpu -f

# Stop the runner (GitHub shows it as Offline and jobs wait in the queue)
sudo ./svc.sh stop

# Start it again
sudo ./svc.sh start
```
