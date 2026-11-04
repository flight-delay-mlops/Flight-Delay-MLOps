# The models train.py can train, picked by name in params.yaml (train.model).
# This file only says what each model is; train.py fits, evaluates and logs them all the same way.
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

    raise ValueError(f"unknown model {name!r} in params.yaml")
