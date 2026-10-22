# Workflows

GitHub Actions runs every `.yml` file in this folder automatically. Each file is
a **workflow**: *when* something happens (a push, a pull request, a button),
*run* these steps on one of our runners. Results show up in the **Actions** tab
on GitHub and as checks on each pull request.

| Workflow | Runs when | What it does |
|---|---|---|
| [ci.yml](ci.yml) | Every PR into `dev` or `main`, every push to them | Checks the code: lint + tests |
| [pipeline.yml](pipeline.yml) | A merge to `main` that changes the pipeline, or by hand | Runs the DVC pipeline and uploads its outputs |

Both run on our **self-hosted runners** (`runs-on: [self-hosted, Linux, X64]`),
not on GitHub's machines, because only machines inside DEI's network can reach
the VMs. See [docs/infrastructure.md](../../docs/infrastructure.md) for the runners.

## ci.yml: "is this code OK?"

The safety net for every change. A PR should only be merged when this is green.

1. **Checkout**: downloads the branch's code onto the runner.
2. **Set up Python 3.11**: the same version we use locally.
3. **Install dependencies** from `requirements-dev.txt` (the project's packages
   plus `pytest` and `ruff`).
4. **Lint** (`ruff check .`): reads the code without running it and fails on
   likely bugs (undefined names, unused imports), style errors and unsorted imports.
5. **Test** (`pytest -v`): runs everything in `tests/`, e.g. that delays are
   labelled correctly and that no outcome column leaks into the features.

It is fast (no data, no VM access needed) and a new push to the same branch
cancels the run still in progress, so there is never a queue of outdated checks.

Run the same checks locally before pushing:

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

## pipeline.yml: "rebuild the data and the model"

Runs `dvc repro`, the same command you would run locally, but on the runner, so
the results come from a clean, known environment instead of someone's laptop.

1. **Checkout, set up Python, install** `requirements.txt`.
2. **Configure DVC**: writes the SSH key from the `DVC_SSH_KEY` secret to a
   temporary file so DVC can log in to VM1, and points the DVC cache to a folder
   on the runner that survives between runs (so only new data is downloaded).
3. **Pull data** (`dvc pull --allow-missing`): downloads the data the current
   `dvc.lock` points to. If some of it isn't on VM1 (e.g. someone committed a new
   `dvc.lock` but forgot `dvc push`), it skips it instead of failing: the next
   step rebuilds it. Missing *raw* data still fails, at the next step, since
   nothing can rebuild it.
4. **Build image, run pipeline**: builds the training image from the root
   `Dockerfile` and runs `dvc repro` inside it, which re-runs only the stages
   whose code, data or `params.yaml` changed (`prepare`, `train`). Training logs
   to MLflow with the `ci` user.
5. **Push outputs** (`dvc push`): uploads the new outputs to VM1.
6. **Report**: the run's summary page shows what changed in `dvc.lock`, and the
   new `dvc.lock` is saved as a downloadable artifact (`dvc-lock`).
7. **Remove SSH key**: deletes the key file, even if an earlier step failed.

Other details:

- It only starts on `main` when a file the pipeline depends on changed (`src/`,
  `dvc.yaml`, `params.yaml`, ...), so docs-only merges don't re-run it.
- Use **Actions → Pipeline → Run workflow** to run it by hand on any branch. Tick
  **Retrain even if nothing changed** to force training (e.g. after a `Dockerfile` change).
- Only one run at a time, and a run is never cancelled, so a `dvc push` is never
  cut off halfway.

### Why CI doesn't check that `dvc.lock` is up to date

A `dvc status` step in CI would fail every PR that changes pipeline code without
a locally regenerated `dvc.lock`, which means training on our own laptops.
We want the opposite: the pipeline runs here, on the runner. The `dvc.lock` it
produces is recorded in the run summary and artifact, and will be logged with
each model in MLflow, so every model can be traced to its exact code and data.

## Setup (once, on GitHub)

| Name | Kind | Value |
|---|---|---|
| `DVC_SSH_KEY` | Secret (Settings → Secrets and variables → Actions) | A private key whose public half is in `~admin/.ssh/authorized_keys` on VM1 |
| `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | Secrets | Login of the `ci` MLflow user (created by an admin in the MLflow UI) |

Create a key just for CI rather than reusing someone's personal one, so it can
be revoked on its own:

```bash
ssh-keygen -t ed25519 -f ci_dvc_key -N "" -C "github-actions dvc"
```

Add `ci_dvc_key.pub` to VM1's `authorized_keys`, paste the contents of
`ci_dvc_key` into the secret, then delete both files from your machine.

## Coming next

| Workflow | When | What |
|---|---|---|
| Training in `pipeline.yml` | When `train.py` exists | Logs to MLflow and registers the model (needs the MLflow credentials from the GitHub Environment) |
| `build.yml` | When `api/Dockerfile` exists | Tests the API, builds the image, pushes it to GHCR |
| `deploy.yml` | When the API works | Pulls the image on the deployment VM, restarts it, checks `/classify` |
| `monitor.yml` | December | Scheduled drift and performance checks, with alerts |
