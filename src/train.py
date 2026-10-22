# Trains the model on the prepared data and logs everything to MLflow.
# For now the models are baselines: always the most common class, or random guesses.
import json

import mlflow
import pandas as pd
import yaml
from dotenv import load_dotenv
from mlflow import MlflowClient
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

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

    # The baseline ignores the features, so only the training labels are loaded:
    # all 9.8M training rows would not fit in the runner VM's 4 GB.
    # pyarrow types keep the validation set small in memory too
    y_train = pd.read_parquet("data/prepared/train.parquet", columns=["label"]).label
    X_val = pd.read_parquet("data/prepared/val.parquet", dtype_backend="pyarrow")
    y_val = X_val.pop("label")

    # most_frequent: always predicts on_time
    # stratified: random guesses, in the same proportions as the training labels
    model = DummyClassifier(strategy=params["strategy"], random_state=params["seed"])
    model.fit(pd.DataFrame(index=y_train.index), y_train)  # no feature columns
    y_pred = model.predict(X_val)
    metrics = evaluate(y_val, y_pred)
    print(json.dumps(metrics, indent=2))

    mlflow.set_experiment(MODEL_NAME)
    with mlflow.start_run(run_name=f"baseline-{params['strategy']}"):
        mlflow.log_params({**params, "train_rows": len(y_train), "val_rows": len(X_val)})
        mlflow.set_tag("data_md5", data_md5())
        mlflow.log_metrics(metrics)
        confusion = pd.crosstab(y_val, y_pred, rownames=["true"], colnames=["predicted"])
        mlflow.log_text(confusion.to_csv(), "confusion_matrix.csv")
        info = mlflow.sklearn.log_model(model, name="model", registered_model_name=MODEL_NAME,
                                        input_example=X_val.head())

    # The newest model is the candidate; promoting it to @production is a separate decision
    MlflowClient().set_registered_model_alias(MODEL_NAME, "staging",
                                              info.registered_model_version)


if __name__ == "__main__":
    main()
