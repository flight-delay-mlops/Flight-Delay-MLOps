from unittest.mock import Mock

import pytest
from mlflow.exceptions import MlflowException

from api.main import app

VALID_PAYLOAD = {"instances": [{"ORIGIN": "DTW"}]}


def classify_with(client, model, payload=VALID_PAYLOAD):
    app.state.model = model
    return client.post("/classify", json=payload)


def test_invalid_request_is_answered_with_422_naming_the_instance_and_field(client):
    response = classify_with(client, Mock(), {"instances": [{}, {"DISTANCE": "far"}]})

    assert response.status_code == 422
    (issue,) = response.json()["detail"]
    assert issue["loc"] == ["body", "instances", 1, "DISTANCE"]
    assert set(issue) == {"loc", "msg", "type"}


@pytest.mark.parametrize(
    "payload",
    [{}, {"instances": []}, {"instances": "DTW"}, {"instances": [5]}],
    ids=["missing", "empty", "not-a-list", "element-not-an-object"],
)
def test_malformed_instances_are_answered_with_422(client, payload):
    response = classify_with(client, Mock(), payload)

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


def test_invalid_json_is_answered_with_422(client):
    app.state.model = Mock()

    response = client.post(
        "/classify", content="{oops", headers={"content-type": "application/json"}
    )

    assert response.status_code == 422


def test_no_model_is_answered_with_503(client):
    response = classify_with(client, None)

    assert response.status_code == 503
    assert response.json() == {"detail": "Model not loaded"}


def test_unexpected_failure_is_answered_with_500_without_internal_details(client):
    model = Mock()
    model.predict.side_effect = ValueError("secret internal detail")

    response = classify_with(client, model)

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "secret" not in response.text


def test_mlflow_failure_that_is_not_a_bad_input_is_answered_with_500(client):
    model = Mock()
    model.predict.side_effect = MlflowException("the registry is down")

    response = classify_with(client, model)

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}


def test_unknown_label_from_the_model_is_answered_with_500(client):
    model = Mock()
    model.predict.return_value = ["late"]

    response = classify_with(client, model)

    assert response.status_code == 500
