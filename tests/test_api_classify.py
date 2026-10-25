from unittest.mock import Mock

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

from api.main import app
from api.schemas import Flight


def classify_with(client, labels, payload):
    model = Mock()
    model.predict.return_value = labels
    app.state.model = model
    return client.post("/classify", json=payload), model


def test_classify_answers_one_label_per_flight_in_order(client):
    labels = ["severe_delay", "on_time", "moderate_delay"]
    payload = {"instances": [{"ORIGIN": "AAA"}, {"ORIGIN": "BBB"}, {"ORIGIN": "CCC"}]}

    response, _ = classify_with(client, labels, payload)

    assert response.status_code == 200
    assert response.json() == {"classifications": labels}


def test_classify_gives_the_model_a_row_per_flight_with_every_field(client):
    payload = {"instances": [{"ORIGIN": "AAA"}, {"ORIGIN": "BBB"}]}

    _, model = classify_with(client, ["on_time", "on_time"], payload)

    (frame,) = model.predict.call_args.args
    assert list(frame.ORIGIN) == ["AAA", "BBB"]
    assert list(frame.columns) == list(Flight.model_fields)


def test_classify_ignores_unknown_fields(client):
    payload = {"instances": [{"ORIGIN": "AAA", "DEP_TIME": 905, "ARR_DELAY": 12.0}]}

    response, model = classify_with(client, ["on_time"], payload)

    (frame,) = model.predict.call_args.args
    assert response.status_code == 200
    assert "DEP_TIME" not in frame.columns


def test_classify_treats_missing_keys_as_null(client):
    _, model = classify_with(client, ["on_time"], {"instances": [{"ORIGIN": "DTW"}]})

    (frame,) = model.predict.call_args.args
    assert frame.TAIL_NUM.isna().all()


def test_classify_returns_only_the_reason_when_the_model_rejects_the_input(client):
    model = Mock()
    model.predict.side_effect = MlflowException(
        "Failed to enforce schema of data '<the whole request>' with schema '<the schema>'. "
        "Error: Model is missing inputs ['DISTANCE'].",
        error_code=INVALID_PARAMETER_VALUE,
    )
    app.state.model = model

    response = client.post("/classify", json={"instances": [{}]})

    assert response.status_code == 422
    assert response.json() == {"detail": "Model is missing inputs ['DISTANCE']."}
