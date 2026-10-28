# Workflows

GitHub Actions runs every `.yml` file in this folder automatically. Each file is
a **workflow**: *when* something happens (a push, a pull request, a button),
*run* these steps on one of our runners. Results show up in the **Actions** tab
on GitHub and as checks on each pull request.

| Workflow | Runs when | What it does |
|---|---|---|
| [ci.yml](ci.yml) | Every PR into `dev` or `main`, every push to them | Checks the code (lint + tests), then builds and smoke-tests the API image; on `main` also publishes it to GHCR |
| [pipeline.yml](pipeline.yml) | A merge to `main` that changes the pipeline, or by hand | Runs the DVC pipeline and uploads its outputs |

Both run on our **self-hosted runners** (`runs-on: [self-hosted, Linux, X64]`),
not on GitHub's machines, because only machines inside DEI's network can reach
the VMs. See [docs/infrastructure.md](../../docs/infrastructure.md) for the runners.

## ci.yml: "is this code OK, and does the API image work?"

The safety net for every change. A PR should only be merged when this is green.
It has two jobs; the second only starts if the first passed (`needs: test`), so
code that fails its tests never gets an image.

### Job 1, `test`

1. **Checkout**: downloads the branch's code onto the runner.
2. **Set up Python 3.11**: the same version we use locally.
3. **Install dependencies** from `requirements-dev.txt` (the project's packages
   plus `pytest` and `ruff`).
4. **Lint** (`ruff check .`): reads the code without running it and fails on
   likely bugs (undefined names, unused imports), style errors and unsorted imports.
5. **Test** (`pytest -v`): runs everything in `tests/`, e.g. that delays are
   labelled correctly and that no outcome column leaks into the features.

It is fast (no data, no VM access needed).

Run the same checks locally before pushing:

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

### Job 2, `api-image`

Packages the inference API as a Docker image ([api/Dockerfile](../../api/Dockerfile),
explained in [api/README.md](../../api/README.md)) and checks that the image
actually starts.

1. **Build image**: `docker build -f api/Dockerfile -t flight-delay-api:ci .`, with
   two labels: the repository (so GitHub links the published image to this repo)
   and the commit it was built from. The runner keeps Docker's layer cache between
   runs, so a rebuild takes seconds unless `api/requirements.txt` changed.
2. **Smoke test**: starts the image with no MLflow access and checks, over HTTP, that:
   - `/health` answers, reporting `"model_loaded": false`,
   - a valid `/classify` request gets **503** (no model, but the app is up),
   - an empty `instances` list gets **422** (input validation works).

   This catches a broken image (missing package, wrong start command, app that
   crashes on startup) before anything is published. It doesn't test predictions:
   that needs a model from MLflow and is the job of the coming deploy/E2E step.
3. **Push to GHCR** (only on pushes to `main` and manual runs, never on PRs):
   tags the image `ghcr.io/<owner>/<repo>/api:<commit SHA>` and pushes it to
   GitHub's container registry. The SHA tag never moves, so every published commit
   can be deployed (or rolled back to) by its SHA. The image name is lowercased
   because image names can't contain capitals. The run's summary shows the full name.
4. **Clean up** (always, even after a failure): removes the smoke-test container
   and the registry tag, logs out of GHCR and deletes dangling images. It keeps
   `flight-delay-api:ci`, whose layers make the next build fast, so the runner's
   disk doesn't fill up with one image per commit.

Why it's built this way:

- **Self-hosted runner**: the deploy step will run on the same VM, and the local
  layer cache makes builds fast.
- **Every PR builds the image** (no path filter): with the cache it costs seconds,
  and it catches problems caused by files outside `api/` too (e.g. the workflow).
- **Cancelling**: a new push to a PR cancels its run in progress, but runs on
  `dev`/`main` are never cancelled, so every merged commit gets its image.
- **Least privilege**: the workflow's token can only read the code; only the
  `api-image` job gets `packages: write`, which it needs to push to GHCR.

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

Publishing the API image needs no secret (the run's own `GITHUB_TOKEN` is used),
but GitHub must allow it:

- **Settings → Actions → General → Workflow permissions**: either setting works,
  since the job asks for `packages: write` itself. In an organization, an org
  admin may also need to allow GitHub Actions to create packages
  (Organization settings → Packages).
- After the first push, the image appears under the repository's **Packages**.
  It is private by default; the deployment VM will need to log in to pull it
  (or make the package public in its **Package settings**).

## Coming next

| Workflow | When | What |
|---|---|---|
| `deploy.yml` | Next | Pulls the image on the deployment VM, restarts it, checks `/classify` with the real model |
| E2E check + promotion | After deploy | Tests the new image with the `@staging` model, then moves `@production` if it isn't worse |
| `monitor.yml` | December | Scheduled drift and performance checks, with alerts |
