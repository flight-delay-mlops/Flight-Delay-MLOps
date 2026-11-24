import pandas as pd

from api.features import FEATURES
from src.models import build_model

FLIGHTS = pd.DataFrame({
    "OP_UNIQUE_CARRIER": ["AA", "DL", "AA", "UA"],
    "ORIGIN": ["JFK", "ATL", "JFK", "ORD"],
    "DEST": ["LAX", "JFK", "ORD", "ATL"],
    "DEP_TIME_BLK": ["0600-0659", "1800-1859", "0600-0659", "2100-2159"],
    "DISTANCE": [2475.0, 760.0, 740.0, 606.0],
    **{name: [5.0, 40.0, 3.0, 20.0] for name in FEATURES},
})
LABELS = ["on_time", "severe_delay", "on_time", "moderate_delay"]


def test_logreg_handles_new_airports_and_missing_distance():
    model = build_model("logreg", seed=42)
    model.fit(FLIGHTS, LABELS)

    # The evaluation data can have airlines and airports that were never in training
    # and the feature store has no rates for them
    new = pd.DataFrame([{"OP_UNIQUE_CARRIER": "ZZ", "ORIGIN": "XXX", "DEST": "YYY",
                         "DEP_TIME_BLK": "0600-0659", "DISTANCE": None,
                         **{name: None for name in FEATURES}}])

    assert model.predict(new)[0] in LABELS
