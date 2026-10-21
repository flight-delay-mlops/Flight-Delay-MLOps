# Infrastructure

Everything runs on two DEI VMs, reachable only through the DEI VPN.

| VM                | IP          | Runs                                                        |
| ----------------- | ----------- | ----------------------------------------------------------- |
| VM1 (4 CPU, 8 GB) | 10.17.0.167 | MLflow server, DVC store                                    |
| VM2 (2 CPU, 4 GB) | 10.17.0.156 | GitHub Actions runner `cpu`, later the API and monitoring |

Both run Ubuntu 24.04 with Docker, set up by `infra/setup_vm.sh`. Log in with `ssh admin@<IP>`.

## MLflow

UI and tracking API at http://10.17.0.167:5000 (login required).

To log from Python, set these variables (see `.env.template`):

```bash
MLFLOW_TRACKING_URI=http://10.17.0.167:5000
MLFLOW_TRACKING_USERNAME=...
MLFLOW_TRACKING_PASSWORD=...
```

The server is defined in `mlflow/`. Runs, users and artifacts live in the Docker volume `mlflow_mlflow-data`, so they survive restarts. To update it, copy `mlflow/` to `~/mlflow` on VM1 and run `docker compose up -d --build` there.

## DVC

Git stores the code and small `.dvc` / `dvc.lock` files with data hashes. The data itself lives in `~/dvc-store` on VM1.

```bash
dvc pull      # get the data for the current commit
dvc repro     # run the pipeline in dvc.yaml (only stages whose inputs changed)
dvc push      # upload new data or pipeline outputs
```

`dvc pull` connects to VM1 over SSH, so you need the VPN and SSH access (your key on VM1, or `dvc remote modify --local vm1 ask_password true`).

## Runners

Self-hosted GitHub Actions runners pick up workflow jobs. GitHub's own runners can't reach the VMs.

| Runner  | Where                     | Notes                           |
| ------- | ------------------------- | ------------------------------- |
| `cpu` | VM2, `~/actions-runner` | systemd service, starts on boot |
| `gpu` | João's computer lol      | offline when the PC is off      |

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
