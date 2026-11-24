# The models train.py can train, picked by name in params.yaml (train.model).
# This file only says what each model is; train.py fits, evaluates and logs them all the same way.
import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns the models use, all raw BTS fields that the API receives
CATEGORIES = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "DEP_TIME_BLK"]
NUMBERS = ["DISTANCE"]
COLUMNS = CATEGORIES + NUMBERS


def build_model(name, seed):
    # Baseline: always predicts the most common class (on_time)
    if name == "most_frequent":
        return DummyClassifier(strategy="most_frequent")

    # Baseline: random guesses, in the same proportions as the training labels
    if name == "stratified":
        return DummyClassifier(strategy="stratified", random_state=seed)

    # Logistic regression. The preprocessing is part of the model, so the API sends raw flights:
    # - each category becomes 0/1 columns; one never seen in training (a new airport) is all zeros
    # - a missing distance becomes the average one, then distances are scaled
    if name == "logreg":
        columns = ColumnTransformer([
            ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORIES),
            ("numbers", make_pipeline(SimpleImputer(), StandardScaler()), NUMBERS),
        ])
        # balanced: the rare delay classes count as much as on_time, which is what macro-F1 rewards
        return make_pipeline(columns, LogisticRegression(class_weight="balanced", max_iter=1000))

    # TensorFlow decision tree (src/fdt.py) and forest of 5 shallower trees (src/forest.py).
    # Same preprocessing as logreg: the splits are linear in the inputs, so distance is scaled
    if name in ("decision_tree", "decision_forest"):
        columns = ColumnTransformer([
            ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORIES),
            ("numbers", make_pipeline(SimpleImputer(), StandardScaler()), NUMBERS),
        ])
        trees, depth = (1, 6) if name == "decision_tree" else (5, 5)
        return make_pipeline(columns, TensorFlowTrees(trees=trees, depth=depth, seed=seed))

    raise ValueError(f"unknown model {name!r} in params.yaml")


class TensorFlowTrees(ClassifierMixin, BaseEstimator):
    """The TensorFlow fair decision tree (trees=1) or forest (trees>1) with fit and predict.

    The fitted weights are kept as numpy arrays and the TensorFlow model is rebuilt to predict,
    so MLflow can save this estimator like any scikit-learn one.
    """

    def __init__(self, trees=1, depth=6, epochs=3, batch_size=1024, learning_rate=0.01, seed=42):
        self.trees = trees
        self.depth = depth
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed

    def fit(self, X, y):
        # Imported here so the other models don't pay for loading TensorFlow
        import tensorflow as tf

        tf.keras.utils.set_random_seed(self.seed)
        self.classes_, labels = np.unique(np.asarray(y), return_inverse=True)
        self.n_features_in_ = X.shape[1]
        # balanced: the rare delay classes count as much as on_time, which is what macro-F1 rewards
        counts = np.bincount(labels)
        sample_weights = (len(labels) / (len(counts) * counts))[labels].astype("float32")

        model = self._new_model()
        optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        @tf.function
        def train_step(batch, batch_labels, batch_weights):
            with tf.GradientTape() as tape:
                logits = model(batch, training=True)[0]
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=batch_labels, logits=logits)
                loss = tf.reduce_sum(losses * batch_weights) / tf.reduce_sum(batch_weights)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        rng = np.random.default_rng(self.seed)
        for _ in range(self.epochs):
            order = rng.permutation(len(labels))
            for start in range(0, len(order), self.batch_size):
                rows = order[start:start + self.batch_size]
                train_step(_dense(X[rows]), labels[rows], sample_weights[rows])

        self.weights_ = [variable.numpy() for variable in model.variables]
        return self

    def predict_proba(self, X):
        model = self._new_model()
        for variable, weights in zip(model.variables, self.weights_):
            variable.assign(weights)
        starts = range(0, X.shape[0], self.batch_size)
        return np.vstack([model(_dense(X[start:start + self.batch_size])).numpy()
                          for start in starts])

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def _new_model(self):
        from src.fdt import FairDecisionTree
        from src.forest import FairDecisionForest

        # compute_mode "log": the forest's default mode applies the softmax twice, which
        # flattens the probabilities (the top one stayed near 0.57, against 0.98 here)
        if self.trees == 1:
            return FairDecisionTree(self.n_features_in_, self.depth, len(self.classes_),
                                    compute_mode="log")
        return FairDecisionForest(self.trees, self.n_features_in_, self.depth,
                                  len(self.classes_), compute_mode="log")


def _dense(rows):
    # The one-hot columns come as a sparse matrix; the trees take small dense batches
    return (rows.toarray() if issparse(rows) else np.asarray(rows)).astype("float32")
