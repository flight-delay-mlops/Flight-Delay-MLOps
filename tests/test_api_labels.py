import sqlite3
from unittest.mock import Mock

from api.main import app


def classify(client, request_id, labels):
    model = Mock()
    model.predict.return_value = labels
    app.state.model = model
    flights = [{"ORIGIN": "JFK"} for _ in labels]
    return client.post("/classify", json={"request_id": request_id, "mode": "live",
                                          "instances": flights})


def test_labels_are_joined_to_the_predictions_and_scored(client, tmp_path):
    classify(client, "r1", ["on_time", "on_time", "severe_delay"])

    response = client.post("/labels", json={
        "request_id": "r1", "mode": "live", "labels": ["on_time", "moderate_delay", "severe_delay"],
        "outcomes": [{"ARR_DELAY": -3.0}, {"ARR_DELAY": 20.0}, {"ARR_DELAY": 75.0}]})

    assert response.status_code == 200
    db = sqlite3.connect(tmp_path / "live.db")
    mode, accuracy = db.execute(
        "SELECT mode, accuracy FROM requests WHERE request_id = 'r1'").fetchone()
    assert mode == "live"
    assert accuracy == 2 / 3


def test_labels_of_an_unknown_request_still_answer_200(client):
    # Anything but a 200 is a failed delivery, which the leaderboard keeps retrying
    response = client.post("/labels", json={"request_id": "never-seen", "mode": "sandbox",
                                            "labels": ["on_time"], "outcomes": [{}]})

    assert response.status_code == 200


def test_classify_accepts_the_leaderboard_request_fields(client):
    response = classify(client, "r2", ["on_time"])

    assert response.status_code == 200
    assert response.json() == {"classifications": ["on_time"]}
