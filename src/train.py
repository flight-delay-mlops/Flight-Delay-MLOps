# Trains the model chosen in params.yaml on the prepared data and logs everything to MLflow.
# The models themselves are in src/models.py.
import json

import mlflow
import pandas as pd
import yaml
from dotenv import load_dotenv
from mlflow import MlflowClient
from sklearn.metrics import accuracy_score, f1_score

from src.models import COLUMNS, build_model
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


def main():
    load_dotenv()  # MLflow address and login from .env (in CI they come from secrets)
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]

    # Only the columns the models use, with pyarrow types to keep memory low:
    # the runner VM has 4 GB. Training uses a random sample of the 9.8M flights
    X_train = pd.read_parquet("data/prepared/train.parquet", columns=COLUMNS + ["label"],
                              dtype_backend="pyarrow")
    X_train = X_train.sample(n=params["sample_rows"], random_state=params["seed"])
    y_train = X_train.pop("label")
    X_val = pd.read_parquet("data/prepared/val.parquet", columns=COLUMNS + ["label"],
                            dtype_backend="pyarrow")
    y_val = X_val.pop("label")

    model = build_model(params["model"], params["seed"])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    metrics = evaluate(y_val, y_pred)
    print(json.dumps(metrics, indent=2))

    mlflow.set_experiment(MODEL_NAME)
    with mlflow.start_run(run_name=params["model"]):
        mlflow.log_params({**params, "train_rows": len(y_train), "val_rows": len(y_val)})
        mlflow.set_tag("data_md5", data_md5())
        mlflow.log_metrics(metrics)
        confusion = pd.crosstab(y_val, y_pred, rownames=["true"], colnames=["predicted"])
        mlflow.log_text(confusion.to_csv(), "confusion_matrix.csv")
        # MLflow saves the model with skops, which only loads types it was told are safe.
        # The one-hot encoder stores a numpy dtype, so that type is declared here
        info = mlflow.sklearn.log_model(model, name="model", registered_model_name=MODEL_NAME,
                                        input_example=X_val.head(),
                                        skops_trusted_types=["numpy.dtype"])

    # The newest model is the candidate; promoting it to @production is a separate decision
    MlflowClient().set_registered_model_alias(MODEL_NAME, "staging",
                                              info.registered_model_version)


if __name__ == "__main__":
    main()
