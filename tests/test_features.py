import pandas as pd
import pytest

from api import features
from src.features import add_offline_features, base_counts


def test_offline_rates_only_use_days_before_the_flight(tmp_path, monkeypatch):
    # Day 1: one flight 60 min late. Day 2: the flight being looked up, itself 100 min late
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "features").mkdir(parents=True)
    for key in features.KEYS:
        values = {"ORIGIN": "JFK", "OP_UNIQUE_CARRIER": "AA", "ROUTE": "JFK-LAX"}
        days = pd.to_datetime(["2023-01-01", "2023-01-02"])
        pd.DataFrame({key: [values[key]] * 2, "FL_DATE": days,
                      f"{key}_avg_delay": [None, 60.0], f"{key}_delayed_share": [None, 1.0]}
                     ).to_parquet(f"data/features/{key}.parquet")
    flight = pd.DataFrame({"ORIGIN": ["JFK"], "DEST": ["LAX"], "OP_UNIQUE_CARRIER": ["AA"],
                           "FL_DATE": ["2023-01-02"]})

    looked_up = add_offline_features(flight)

    assert looked_up.ORIGIN_avg_delay[0] == 60.0


def test_live_outcomes_update_the_online_rates(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    flights = pd.DataFrame({"ORIGIN": ["JFK", "JFK"], "DEST": ["LAX", "ORD"],
                            "OP_UNIQUE_CARRIER": ["AA", "AA"], "FL_DATE": ["2024-02-01"] * 2})
    assert features.add_online_features(flights).ORIGIN_avg_delay.isna().all()

    features.learn_outcomes(flights, [{"ARR_DELAY": 30.0}, {"ARR_DELAY": -10.0}])

    looked_up = features.add_online_features(flights)
    assert looked_up.ORIGIN_avg_delay[0] == 10.0
    assert looked_up.ORIGIN_delayed_share[0] == 0.5
    assert looked_up.ROUTE_avg_delay.tolist() == [30.0, -10.0]


def test_offline_rates_keep_repeated_flights_and_ignore_the_future(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "features").mkdir(parents=True)
    for key, value in {"ORIGIN": "JFK", "OP_UNIQUE_CARRIER": "AA", "ROUTE": "JFK-LAX"}.items():
        pd.DataFrame({key: [value] * 3,
                      "FL_DATE": pd.to_datetime(["2023-01-30", "2023-02-01", "2023-02-02"]),
                      f"{key}_avg_delay": [10.0, 20.0, 99.0],
                      f"{key}_delayed_share": [0.1, 0.2, 0.9]}
                     ).to_parquet(f"data/features/{key}.parquet")
    # Two equal flights, another month, and an airport the store doesn't know
    flights = pd.DataFrame({"ORIGIN": ["JFK", "JFK", "JFK", "XXX"], "DEST": ["LAX"] * 4,
                            "OP_UNIQUE_CARRIER": ["AA"] * 4,
                            "FL_DATE": ["2023-02-01", "2023-02-01", "2023-01-30", "2023-02-01"]})

    looked_up = add_offline_features(flights)

    assert len(looked_up) == 4
    assert looked_up.ORIGIN_avg_delay[:3].tolist() == [20.0, 20.0, 10.0]
    assert pd.isna(looked_up.ORIGIN_avg_delay[3])
    assert looked_up.OP_UNIQUE_CARRIER_delayed_share.tolist() == [0.2, 0.2, 0.1, 0.2]


@pytest.fixture
def base_rates(monkeypatch, tmp_path):
    # 100 flights of JFK, AA and JFK-LAX, averaging 20 min late, and 25% of them 15+ min late
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    path = tmp_path / "base_rates.parquet"
    pd.DataFrame([{"key": key, "value": value, "flights": 100, "delay": 2000.0, "delayed": 25}
                  for key, value in [("ORIGIN", "JFK"), ("OP_UNIQUE_CARRIER", "AA"),
                                     ("ROUTE", "JFK-LAX")]]).to_parquet(path)
    monkeypatch.setattr(features.mlflow.artifacts, "download_artifacts", lambda **_: str(path))
    features.load_base_rates("run")


FLIGHT = pd.DataFrame({"ORIGIN": ["JFK"], "DEST": ["LAX"], "OP_UNIQUE_CARRIER": ["AA"],
                       "FL_DATE": ["2024-02-01"]})


def test_online_store_starts_from_the_base_rates(base_rates):
    looked_up = features.add_online_features(FLIGHT)

    assert looked_up.ORIGIN_avg_delay[0] == 20.0
    assert looked_up.ROUTE_delayed_share[0] == 0.25


def test_live_outcomes_are_blended_with_the_base_rates(base_rates):
    new = FLIGHT.assign(ORIGIN="XXX")
    flights = pd.concat([FLIGHT, new], ignore_index=True)

    features.learn_outcomes(flights, [{"ARR_DELAY": 122.0}, {"ARR_DELAY": 122.0}])
    features.load_base_rates("run")  # a restart doesn't add the base a second time

    looked_up = features.add_online_features(flights)
    # One live flight against 50 base ones; a new airport starts from the average
    assert looked_up.ORIGIN_avg_delay.tolist() == pytest.approx([22.0, 22.0])
    assert looked_up.OP_UNIQUE_CARRIER_avg_delay[0] == pytest.approx((50 * 20 + 244) / 52)


def test_base_counts_are_the_last_30_days():
    counts = pd.DataFrame({"ORIGIN": ["JFK"] * 3, "flights": [1, 2, 3], "delay": [1.0, 2.0, 3.0],
                           "delayed": [0, 1, 1],
                           "FL_DATE": pd.to_datetime(["2023-12-01", "2023-12-02", "2023-12-31"])})

    base = base_counts(counts, "ORIGIN")

    assert base[["key", "value", "flights", "delay", "delayed"]].values.tolist() == [
        ["ORIGIN", "JFK", 5, 5.0, 2]]
