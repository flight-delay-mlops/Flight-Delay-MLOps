import mlflow
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score

from api.features import FEATURES
from src.models import TensorFlowTrees, WeightedDecision, _macro_f1, build_model
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


@pytest.mark.parametrize("compute_mode", ["log", "default"])
def test_forest_gives_the_mean_of_its_trees_leaf_probabilities(compute_mode):
    import tensorflow as tf

    from src.forest import FairDecisionForest

    tf.keras.utils.set_random_seed(0)
    flights = tf.random.normal([16, 4])
    forest = FairDecisionForest(3, 4, 3, 3, compute_mode=compute_mode)

    expected = np.mean([tf.matmul(tree(flights, training=True)[2], tf.nn.softmax(tree.theta))
                        for tree in forest.layers], axis=0)

    np.testing.assert_allclose(forest(flights).numpy(), expected, atol=1e-5)
    # What training's cross-entropy uses: log-probabilities of the same distribution
    np.testing.assert_allclose(np.exp(forest(flights, training=True)[0]), expected, atol=1e-5)


class FixedProbabilities:
    # A classifier that answers the probabilities it is given
    classes_ = np.array(["moderate_delay", "on_time", "severe_delay"])

    def predict_proba(self, X):
        return np.asarray(X)


def test_decision_weights_find_the_rare_classes():
    # Mostly on time, and the delayed flights are never the most likely class
    X = np.array([[0.15, 0.80, 0.05]] * 80 + [[0.30, 0.60, 0.10]] * 14 + [[0.15, 0.60, 0.25]] * 6)
    y = np.array(["on_time"] * 80 + ["moderate_delay"] * 14 + ["severe_delay"] * 6)

    model = WeightedDecision(FixedProbabilities()).fit(X, y)

    assert (FixedProbabilities.classes_[X.argmax(axis=1)] == "on_time").all()
    assert (model.predict(X) == y).all()
    assert model.weights_[1] == 1  # on_time is the reference
    np.testing.assert_array_equal(model.predict_proba(X), X)


def test_fast_macro_f1_matches_scikit_learn():
    rng = np.random.default_rng(0)
    labels, predicted = rng.integers(0, 3, 500), rng.integers(0, 3, 500)

    assert _macro_f1(labels, predicted, 3) == pytest.approx(
        f1_score(labels, predicted, average="macro"))


@pytest.mark.parametrize("name", ["logreg", "decision_forest"])
def test_decision_weights_are_saved_with_the_model(name, tmp_path):
    model = WeightedDecision(build_model(name, seed=42).fit(FLIGHTS, LABELS))
    model.fit(FLIGHTS, LABELS)

    mlflow.sklearn.save_model(model, str(tmp_path / "model"), input_example=FLIGHTS,
                              skops_trusted_types=SKOPS_TRUSTED_TYPES)
    loaded = mlflow.pyfunc.load_model(str(tmp_path / "model"))

    assert list(loaded.predict(FLIGHTS)) == list(model.predict(FLIGHTS))
    np.testing.assert_array_equal(loaded.get_raw_model().weights_, model.weights_)


def test_weighted_gradients_raise_the_probability_of_the_rare_class():
    # The features say nothing, so the model learns how often each class comes, times its weight
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 3)).astype("float32")
    y = np.array(["on_time"] * 800 + ["severe_delay"] * 200)

    def severe_probability(power):
        model = TensorFlowTrees(trees=2, depth=2, epochs=40, batch_size=100, learning_rate=0.02,
                                class_weight_power=power).fit(X, y)
        return model.predict_proba(X)[:, 1].mean()

    # Weights 1 and 4 at power 1, 1 and 2 at power 0.5
    assert severe_probability(0) == pytest.approx(0.2, abs=0.05)
    assert severe_probability(0.5) == pytest.approx(1 / 3, abs=0.07)
    assert severe_probability(1) == pytest.approx(0.5, abs=0.08)
