# Flight-Delay-MLOps

End-to-end MLOps pipeline that classifies US domestic flights by arrival delay
severity (`on_time`, `moderate_delay`, `severe_delay`), using BTS On-Time
Performance data from 2022–2023. The assignment is in
[docs/statement/](docs/statement/).

- VMs, MLflow, DVC and runners: [docs/infrastructure.md](docs/infrastructure.md)
- Branches, commits and pull requests: [CONTRIBUTING.md](CONTRIBUTING.md)

## Repository structure

```
.github/workflows/  CI/CD pipelines (GitHub Actions)
api/                Inference service: FastAPI app and its Dockerfile
data/               Managed by DVC, never committed to Git directly
  raw.dvc           Pointer to the raw BTS files (data/raw/)
  prepared/         Output of the prepare stage (train/val Parquet)
docs/               Project documentation, the assignment and the report
infra/              Everything deployed by hand on the VMs
  setup_vm.sh       One-time VM setup (Docker, disk)
  mlflow/           Docker setup of the MLflow server on VM1
src/                Data and training pipeline code (run by DVC stages)
tests/              Automated tests (pytest), run by CI
dvc.yaml, dvc.lock  Pipeline definition and the exact versions it last ran with
params.yaml         Pipeline settings, tracked by DVC
```

Why it is split this way:

- **`src/` vs `api/`**: training and serving have different lifecycles and
  dependencies. `src/` runs in batch through DVC to produce a model; `api/` is a
  long-running service packaged as its own Docker image. Feature engineering
  lives in `src/` and `api/` imports it, so serving applies exactly the same
  transformations as training (no training/serving skew).
- **`infra/` apart from the ML code**: it describes the machines and the
  services they host, not the model. It changes rarely, is deployed by hand on
  the VMs, and should not trigger training when edited.
- **`data/` handled by DVC**: Git only stores small `.dvc` pointers and
  `dvc.lock`; the files themselves are in the DVC remote. Checking out any commit
  and running `dvc pull` gives the exact data that commit used.
- **`params.yaml` separate from code**: DVC tracks it as a stage input, so
  changing a setting (e.g. the validation split date) marks the affected stages
  to re-run, and the value is recorded in `dvc.lock`.
- **`tests/` at the top level**: tests cover both `src/` and `api/`, and CI runs
  them all with a single `pytest`.
- **`docs/`**: everything meant to be read rather than run, kept out of the way
  of the code.
- **`.github/workflows/`**: the path GitHub Actions requires.
- **`README.md` and `CONTRIBUTING.md` at the root**: GitHub shows them on the
  repository page and when opening a pull request.

## Getting started

Requires access to the DEI network (or VPN) and SSH access to VM1.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
dvc pull     # download data/raw and data/prepared from VM1
dvc repro    # re-run any pipeline stage whose inputs changed
```

To log to MLflow, copy `.env.template` to `.env` (git-ignored) and fill in your
MLflow credentials.
