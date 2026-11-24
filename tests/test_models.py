import mlflow
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from api.features import FEATURES
from src.models import TensorFlowTrees, build_model
from src.train import SKOPS_TRUSTED_TYPES

FLIGHTS = pd.DataFrame({
    "OP_UNIQUE_CARRIER": ["AA", "DL", "AA", "UA"],
    "ORIGIN": ["JFK", "ATL", "JFK", "ORD"],
    "DEST": ["LAX", "JFK", "ORD", "ATL"],
    "DEP_TIME_BLK": ["0600-0659", "1800-1859", "0600-0659", "2100-2159"],
    "DISTANCE": [2475.0, 760.0, 740.0, 606.0],
    **{name: [5.0, 40.0, 3.0, 20.0] for name in FEATURES},
})
LABELS = ["on_time", "severe_delay", "on_time", "moderate_delay"]


@pytest.mark.parametrize("name", ["logreg", "decision_tree", "decision_forest"])
def test_models_handle_new_airports_and_missing_distance(name):
    model = build_model(name, seed=42)
    model.fit(FLIGHTS, LABELS)

    # The evaluation data can have airlines and airports that were never in training
    # and the feature store has no rates for them
    new = pd.DataFrame([{"OP_UNIQUE_CARRIER": "ZZ", "ORIGIN": "XXX", "DEST": "YYY",
                         "DEP_TIME_BLK": "0600-0659", "DISTANCE": None,
                         **{name: None for name in FEATURES}}])

    assert model.predict(new)[0] in LABELS


@pytest.mark.parametrize("name", ["decision_tree", "decision_forest"])
def test_tensorflow_models_give_the_same_predictions_after_saving_and_loading(name, tmp_path):
    model = build_model(name, seed=42)
    model.fit(FLIGHTS, LABELS)

    mlflow.sklearn.save_model(model, str(tmp_path / "model"), input_example=FLIGHTS,
                              skops_trusted_types=SKOPS_TRUSTED_TYPES)
    loaded = mlflow.pyfunc.load_model(str(tmp_path / "model"))

    assert list(loaded.predict(FLIGHTS)) == list(model.predict(FLIGHTS))


@pytest.mark.parametrize("trees", [1, 3])
@pytest.mark.parametrize("as_matrix", [np.asarray, csr_matrix], ids=["dense", "sparse"])
def test_tensorflow_trees_learn_a_simple_rule(trees, as_matrix):
    rng = np.random.default_rng(0)
    features = rng.integers(0, 2, size=(400, 4)).astype("float32")
    labels = np.where(features[:, 0] == 1, "severe_delay", "on_time")

    model = TensorFlowTrees(trees=trees, depth=2, epochs=60, batch_size=100, learning_rate=0.05)
    model.fit(as_matrix(features), labels)

    assert (model.predict(as_matrix(features)) == labels).mean() > 0.95
    assert set(model.classes_) == {"on_time", "severe_delay"}
