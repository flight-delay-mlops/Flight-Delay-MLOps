# Trains the model chosen in params.yaml on the prepared data and logs everything to MLflow.
# The models themselves are in src/models.py.
import json

import mlflow
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import accuracy_score, f1_score

from src.features import add_offline_features
from src.models import CATEGORIES, COLUMNS, build_model
from src.prepare import LABELS

MODEL_NAME = "flight-delay"


def evaluate(y_true, y_pred):
    # Macro-F1 weighs the three classes equally, so a model that ignores
    # the delays scores badly even with a high accuracy
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": per_class.mean(),
    }
    for label, f1 in zip(LABELS, per_class):
        metrics[f"f1_{label}"] = f1
    return metrics


def data_md5():
    # Hash of data/prepared as recorded by DVC, to trace each run back to its exact data
    with open("dvc.lock") as f:
        lock = yaml.safe_load(f)
    return lock["stages"]["prepare"]["outs"][0]["md5"]


def load_flights(split):
    # Only the columns the models need, with pyarrow types to keep memory low
    return pd.read_parquet(f"data/prepared/{split}.parquet", dtype_backend="pyarrow",
                           columns=CATEGORIES + ["DISTANCE", "FL_DATE", "label"])


def production_macro_f1(client):
    # Macro-F1 of the model the API serves now, or None before anything was promoted
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, "production")
    except MlflowException:
        return None
    return client.get_run(version.run_id).data.metrics["macro_f1"]


def main():
    load_dotenv()  # MLflow address and login from .env (in CI they come from secrets)
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]

    # Fit on a random sample of the 9.8M training flights, then free it before loading the
    # 3.5M validation flights: the runner VM has 4 GB. Each flight gets the delay rates from
    # the offline feature store, as they were before its date
    train = load_flights("train").sample(n=params["sample_rows"], random_state=params["seed"])
    train = add_offline_features(train)
    train_rows = len(train)
    model = build_model(params["model"], params["seed"])
    model.fit(train[COLUMNS], train.label)
    del train

    val = add_offline_features(load_flights("val"))
    X_val, y_val = val[COLUMNS], val.label
    # In chunks of 500k flights, which keeps memory low too
    y_pred = np.concatenate([model.predict(X_val[start:start + 500_000])
                             for start in range(0, len(X_val), 500_000)])
    metrics = evaluate(y_val, y_pred)
    print(json.dumps(metrics, indent=2))

    mlflow.set_experiment(MODEL_NAME)
    with mlflow.start_run(run_name=params["model"]):
        mlflow.log_params({**params, "train_rows": train_rows, "val_rows": len(y_val)})
        mlflow.set_tag("data_md5", data_md5())
        mlflow.log_metrics(metrics)
        confusion = pd.crosstab(y_val, y_pred, rownames=["true"], colnames=["predicted"])
        mlflow.log_text(confusion.to_csv(), "confusion_matrix.csv")
        # MLflow saves the model with skops, which only loads types it was told are safe.
        # The one-hot encoder stores a numpy dtype, so that type is declared here
        info = mlflow.sklearn.log_model(model, name="model", registered_model_name=MODEL_NAME,
                                        input_example=X_val.head(),
                                        skops_trusted_types=["numpy.dtype"])

    # Quality gate: the new model only becomes the candidate (@staging) if it beats the model
    # in production on the validation set. Every version stays in the registry, tagged with
    # whether it passed. Promoting @staging to @production is a separate step (promote.py)
    client = MlflowClient()
    version = info.registered_model_version
    best = production_macro_f1(client)
    passed = best is None or metrics["macro_f1"] > best
    client.set_model_version_tag(MODEL_NAME, version, "quality_gate",
                                 "passed" if passed else "failed")
    if passed:
        client.set_registered_model_alias(MODEL_NAME, "staging", version)
        print(f"Version {version} passed the quality gate and is now @staging")
    else:
        print(f"Version {version} (macro-F1 {metrics['macro_f1']:.3f}) doesn't beat "
              f"@production ({best:.3f}), so @staging stays where it was")


if __name__ == "__main__":
    main()
