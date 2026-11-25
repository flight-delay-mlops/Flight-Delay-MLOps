import pandas as pd
import pytest

from src.train import choose_flights, evaluate, split_for_tuning


def test_evaluate_perfect_predictions():
    y = pd.Series(["on_time", "moderate_delay", "severe_delay"])

    metrics = evaluate(y, y)

    assert metrics["accuracy"] == 1
    assert metrics["macro_f1"] == 1


def test_evaluate_always_on_time():
    # 8 of 10 flights on time: accuracy looks good, macro-F1 shows the delays are missed
    y_true = pd.Series(["on_time"] * 8 + ["moderate_delay", "severe_delay"])
    y_pred = pd.Series(["on_time"] * 10)

    metrics = evaluate(y_true, y_pred)

    assert metrics["accuracy"] == pytest.approx(0.8)
    assert metrics["f1_moderate_delay"] == 0
    assert metrics["f1_severe_delay"] == 0
    assert metrics["macro_f1"] == pytest.approx(metrics["f1_on_time"] / 3)


def test_tuning_flights_come_after_the_ones_the_model_is_fitted_on():
    flights = pd.DataFrame({"FL_DATE": pd.date_range("2023-01-01", "2023-06-30")})
    params = {"tune_start": "2023-04-01", "sample_rows": 20, "tune_rows": 10, "seed": 42}

    fit, tune = split_for_tuning(flights, params)

    assert (len(fit), len(tune)) == (20, 10)
    assert fit.FL_DATE.max() < pd.Timestamp("2023-04-01") <= tune.FL_DATE.min()


@pytest.mark.parametrize("imbalance, model, has_tune", [
    ("gradient_weights", "decision_forest", False),
    ("decision_weights", "decision_forest", True),
    ("decision_weights", "most_frequent", False),
])
def test_only_decision_weights_keep_flights_for_tuning(imbalance, model, has_tune):
    flights = pd.DataFrame({"FL_DATE": pd.date_range("2023-01-01", "2023-06-30")})
    params = {"imbalance": imbalance, "model": model, "tune_start": "2023-04-01",
              "sample_rows": 80, "tune_rows": 10, "seed": 42}

    fit, tune = choose_flights(flights, params)

    assert len(fit) == 80
    assert (tune is not None) == has_tune
    # Without tuning, the last months also train the model
    assert (fit.FL_DATE.max() >= pd.Timestamp("2023-04-01")) != has_tune
