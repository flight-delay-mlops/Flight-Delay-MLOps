import pandas as pd
import pytest

from src.train import evaluate


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
