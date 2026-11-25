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

from src.features import BASE_PATH, add_offline_features
from src.models import BASELINES, CATEGORIES, COLUMNS, WeightedDecision, build_model
from src.prepare import LABELS

MODEL_NAME = "flight-delay"

# The two ways to deal with the rare delay classes (src/README.md)
IMBALANCE = ("gradient_weights", "decision_weights")

# MLflow saves the model with skops, which only loads types it was told are safe: the numpy
# dtype that the one-hot encoder stores, the TensorFlow trees and the decision weights
SKOPS_TRUSTED_TYPES = ["numpy.dtype", "src.models.TensorFlowTrees",
                       "src.models.WeightedDecision"]


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


def split_for_tuning(flights, params):
    # The last months of the training data choose the decision weights, so the model is fitted
    # only on flights before them. The validation set stays out of it: it feeds the quality gate
    is_tune = pd.to_datetime(flights.FL_DATE) >= params["tune_start"]
    fit, tune = flights[~is_tune], flights[is_tune]
    return (fit.sample(n=params["sample_rows"], random_state=params["seed"]),
            tune.sample(n=params["tune_rows"], random_state=params["seed"]))


def choose_flights(flights, params):
    # Flights to fit the model on, and to tune the decision weights on (None if not used).
    # Weighted gradients need no tuning flights, so the model can learn from all the months
    if params["imbalance"] == "decision_weights" and params["model"] not in BASELINES:
        return split_for_tuning(flights, params)
    return flights.sample(n=params["sample_rows"], random_state=params["seed"]), None


def predict_proba(model, X):
    # In chunks of 500k flights, which keeps memory low
    return np.concatenate([model.predict_proba(X[start:start + 500_000])
                           for start in range(0, len(X), 500_000)])


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

    # Fit on a random sample of the training flights, then free it before loading the
    # 3.5M validation flights: the runner VM has 4 GB. Each flight gets the delay rates from
    # the offline feature store, as they were before its date
    if params["imbalance"] not in IMBALANCE:
        raise ValueError(f"unknown imbalance {params['imbalance']!r} in params.yaml")
    train, tune = choose_flights(load_flights("train"), params)
    train = add_offline_features(train)
    train_rows = len(train)
    power = params["class_weight_power"] if params["imbalance"] == "gradient_weights" else 0.0
    model = build_model(params["model"], params["seed"], power)
    model.fit(train[COLUMNS], train.label)
    del train
    if tune is not None:
        tune = add_offline_features(tune)
        model = WeightedDecision(model).fit(tune[COLUMNS], tune.label)
    del tune

    val = add_offline_features(load_flights("val"))
    X_val, y_val = val[COLUMNS], val.label
    probabilities = predict_proba(model, X_val)
    y_pred = model.classes_[probabilities.argmax(axis=1)]
    metrics = evaluate(y_val, y_pred)
    if isinstance(model, WeightedDecision):
        # The untuned score shows how much the decision weights are worth
        untuned = metrics["macro_f1"]
        y_pred = model.classes_[(probabilities * model.weights_).argmax(axis=1)]
        metrics = {**evaluate(y_val, y_pred), "macro_f1_untuned": untuned}
        for label, weight in zip(model.classes_, model.weights_):
            metrics[f"decision_weight_{label}"] = weight
    print(json.dumps(metrics, indent=2))

    mlflow.set_experiment(MODEL_NAME)
    with mlflow.start_run(run_name=params["model"]):
        mlflow.log_params({**params, "train_rows": train_rows, "val_rows": len(y_val)})
        mlflow.set_tag("data_md5", data_md5())
        mlflow.log_metrics(metrics)
        confusion = pd.crosstab(y_val, y_pred, rownames=["true"], colnames=["predicted"])
        mlflow.log_text(confusion.to_csv(), "confusion_matrix.csv")
        # Where the API's feature store starts from; it downloads it with the model
        mlflow.log_artifact(BASE_PATH)
        # code_paths: the saved model carries src/ with it. The TensorFlow trees need fdt.py and
        # forest.py to load, and the API image only has api/
        info = mlflow.sklearn.log_model(model, name="model", registered_model_name=MODEL_NAME,
                                        input_example=X_val.head(),
                                        skops_trusted_types=SKOPS_TRUSTED_TYPES,
                                        code_paths=["src"])

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
