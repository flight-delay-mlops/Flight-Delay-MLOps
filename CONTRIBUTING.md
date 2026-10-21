# Contributing

## Branches

```
main  ◀── PR ── dev  ◀── PR ── feat/... | fix/... | exp/... | chore/...
```

| Branch | Purpose | Why |
|---|---|---|
| `main` | Deploy!. Every merge here retrains, packages and deploys the model. | Keeps the costly pipeline from running on every push. |
| `dev` | Integration of finished work. CI runs the tests on every PR into it. | Lets everyone's work be combined and tested together before it reaches production.|
| `feat/<name>` | A new feature, e.g. `feat/train-stage`. | One branch per task keeps PRs small and reviewable.|
| `fix/<name>` | A bug fix, e.g. `fix/fl-date-format`. | Separating fixes from features makes the history easy to read. |
| `exp/<name>` | A modelling experiment, e.g. `exp/lightgbm-tail-num`. | Experiments may never be merged. Their results live in MLflow, and the branch keeps the exact code that produced each run. |
| `chore/<name>` | Infrastructure, CI, docs, dependencies, e.g. `chore/repo-structure`. | Work that changes no model behaviour. |

Rules:

- Branch from `dev`, open the PR into `dev`.
- Never push directly to `main` or `dev`; at least one other member reviews each PR.
- Names in lowercase with hyphens.
- Delete the branch after merging.

## Commits

- Short, imperative summary of what changed: `Add prepare stage: filter flights, label delays`.
- One logical change per commit; commit often.
- Never commit `.env`, credentials or data files (data goes through `dvc add` / `dvc push`).


