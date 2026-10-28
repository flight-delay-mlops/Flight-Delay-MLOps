# Workflows

GitHub Actions runs every `.yml` file in this folder automatically. Each file is
a **workflow**: *when* something happens (a push, a pull request, a button),
*run* these steps on one of our runners. Results show up in the **Actions** tab
on GitHub and as checks on each pull request.

| Workflow | Runs when | What it does |
|---|---|---|
| [ci.yml](ci.yml) | Every PR into `dev` or `main`, every push to them | Checks the code (lint + tests), then builds and smoke-tests the API image; on `main` also publishes it to GHCR and deploys it |
| [train.yml](train.yml) | A merge to `main` that changes the pipeline, or by hand | Runs the DVC pipeline (prepare + train) and registers the new model in MLflow |
| [deploy.yml](deploy.yml) | Called by `ci.yml` on `main`, or by hand | Runs an API image on VM2 with the `@production` model, checks it, rolls back if it fails |

All run on our **self-hosted runners**, not on GitHub's machines, because only
machines inside DEI's network can reach the VMs. See
[docs/infrastructure.md](../../docs/infrastructure.md) for the runners.

### Naming: CI, CT and CD

Each file is named after what it does, the usual convention (lowercase, one word):

| File | Practice | Ends with |
|---|---|---|
| `ci.yml` | **Continuous Integration**: is this change OK to merge? | Tested code and an image that builds and starts |
| `train.yml` | **Continuous Training**: rebuild the data and the model | A new model version in MLflow (`@staging`) |
| `ci.yml` (on `main`) | **Continuous Delivery**: is a version *ready* to deploy? | An image in GHCR tagged with its commit SHA |
| `deploy.yml` | **Continuous Deployment**: is the new version *running*? | The API live on VM2 |

"CD" means both delivery (ready to ship) and deployment (actually shipped).

## ci.yml: "is this code OK, and does the API image work?"

The safety net for every change. A PR should only be merged when this is green.
It has three jobs, each starting only if the one before passed (`needs:`), so
code that fails its tests never gets an image, and an image that doesn't start
is never deployed:

```
test ──▶ api-image ──▶ deploy (main only, runs deploy.yml)
```

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

   "No MLflow access" means pointing `MLFLOW_TRACKING_URI` at an address where nothing
   answers, with `MLFLOW_HTTP_REQUEST_MAX_RETRIES=0`, so loading the model fails
   within a second. Leaving the variable unset doesn't work: MLflow then tries to
   create a local database (`mlflow.db`) in `/app`, which the container's user
   can't write. It retries for about 100 s before giving up, and the app answers
   nothing until then, so the test timed out after 60 s.
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

### Job 3, `deploy`

Only on pushes to `main`. It doesn't have steps of its own: it calls
[deploy.yml](deploy.yml) (a *reusable workflow*) with this commit's SHA, so the
image that was just tested and published is the one deployed. See
[deploy.yml](#deployyml-is-the-new-version-running) below.

## train.yml: "rebuild the data and the model"

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
- Use **Actions → Train → Run workflow** to run it by hand on any branch. Tick
  **Retrain even if nothing changed** to force training (e.g. after a `Dockerfile` change).
- Only one run at a time, and a run is never cancelled, so a `dvc push` is never
  cut off halfway.

### Why CI doesn't check that `dvc.lock` is up to date

A `dvc status` step in CI would fail every PR that changes pipeline code without
a locally regenerated `dvc.lock`, which means training on our own laptops.
We want the opposite: the pipeline runs here, on the runner. The `dvc.lock` it
produces is recorded in the run summary and artifact, and will be logged with
each model in MLflow, so every model can be traced to its exact code and data.

## deploy.yml: "is the new version running?"

Runs a published API image on VM2, as the container `flight-delay-api` on port 8000,
serving the model that MLflow's `@production` alias points to. The endpoint is
**http://10.17.0.156:8000/classify**.

When it runs:

- **After every merge into `main`**: `ci.yml` calls it once the image is published.
- **By hand** (Actions → Deploy → Run workflow): leave the SHA empty to redeploy
  the latest commit of the chosen branch, e.g. after moving `@production` in MLflow
  (the API only loads the model at startup). Give an older commit's SHA to roll
  back to that version.

Steps:

1. **Pull image** `ghcr.io/<owner>/<repo>/api:<SHA>` from GHCR, logged in with
   the run's own token. Fails clearly if that commit has no published image.
2. **Remember the running version**, to be able to put it back.
3. **Start the new version**: removes the old container and starts the new one
   with `--restart unless-stopped` (back up by itself after a crash or reboot).
   The MLflow address, login and `MODEL_URI` are passed as environment variables.
   The swap leaves the API down for a few seconds.
4. **Check it works**: waits up to 2 minutes for `/health`, then sends a real
   flight to `/classify`, which must succeed. That one request catches every
   problem: without a model the API answers 503, and a label other than the three
   is a 500. If it fails, the run shows the API's answer (e.g.
   `{"detail":"Model not loaded"}`) and the container's log.
5. **Roll back** (only if the check failed and something was running before):
   starts the previous image again, so a bad deployment doesn't leave the
   endpoint broken. The run still shows as failed.
6. **Clean up** (always): logs out of GHCR and keeps only the two newest API
   images on the VM (the running one and the one before, for rollbacks).

The deployed commit and a link to the API are shown on the run page and under the
repo's **Deployments**, through the `production` environment.

Why it's built this way:

- **Reusable workflow instead of `workflow_run`**: the tutorial chains workflows
  with `workflow_run`, but there `github.sha` is the latest commit of the default
  branch, not the commit that was tested, so it can deploy the wrong image.
  Calling `deploy.yml` from `ci.yml` passes the exact SHA.
- **Runs on VM2 only**: `runs-on` includes the `deploy` label, which only VM2's
  runner has. "Deploying" means starting the container on the machine the job
  runs on, so it must not run on another runner.
- **GitHub environment `production`**: every deployment is listed under the
  repo's **Deployments**, with its time, commit and a link to the API. Protection
  rules can be added there later (e.g. require an approval before deploying).
- **Health check and rollback**: the tutorial's deploy starts the container and
  stops there; here a deployment only counts as successful once the API answers
  a real request, and a failed one is undone.

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
  It is private by default; `deploy.yml` logs in with the run's token to pull it.

Deploying needs:

- **The `deploy` label on VM2's runner**: Settings → Actions → Runners → `cpu` →
  edit labels → add `deploy`. Without it the deploy job waits forever for a runner.
- **A model with the `@production` alias in MLflow** (Models → flight-delay →
  pick a version → add the alias). Without it the API starts with no model, the
  check fails and the deployment is rolled back (or, on the very first deployment,
  left running with `/classify` answering 503).
- Nothing else: the `production` environment is created automatically on the
  first deployment, and the MLflow login reuses the secrets above.

## Coming next

| Workflow | When | What |
|---|---|---|
| E2E check + promotion | Next | Tests the new image with the `@staging` model, then moves `@production` if it isn't worse, and redeploys |
| `monitor.yml` | December | Scheduled drift and performance checks, with alerts |
