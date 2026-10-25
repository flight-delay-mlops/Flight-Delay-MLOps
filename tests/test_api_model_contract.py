import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier

from api.main import app
from api.schemas import ClassifyRequest

# What the registered model expects: MLflow checks the input of predict() against this signature
COMPLETE_ROW = {
    "FL_DATE": "2023-07-02",
    "OP_UNIQUE_CARRIER": "9E",
    "OP_CARRIER_FL_NUM": 4900,
    "TAIL_NUM": "N307PQ",
    "ORIGIN": "DTW",
    "ORIGIN_STATE_ABR": "MI",
    "DEST": "DSM",
    "DEST_STATE_ABR": "IA",
    "CRS_DEP_TIME": 900,
    "DEP_TIME_BLK": "0900-0959",
    "CRS_ARR_TIME": 1042,
    "ARR_TIME_BLK": "1000-1059",
    "CRS_ELAPSED_TIME": 102.0,
    "DISTANCE": 528.0,
}


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    example = pd.DataFrame([COMPLETE_ROW])
    example["FL_DATE"] = pd.to_datetime(example["FL_DATE"])
    signature = infer_signature(example, pd.Series(["on_time"]))
    model = DummyClassifier(strategy="most_frequent")
    model.fit(pd.DataFrame(index=range(3)), ["on_time"] * 3)
    path = str(tmp_path_factory.mktemp("model") / "model")
    mlflow.sklearn.save_model(model, path, signature=signature)
    return mlflow.pyfunc.load_model(path)


def frame_of(**overrides):
    return ClassifyRequest(instances=[{**COMPLETE_ROW, **overrides}]).to_dataframe()


def test_model_classifies_a_complete_flight(tiny_model):
    assert list(tiny_model.predict(frame_of())) == ["on_time"]


@pytest.mark.parametrize("field", ["TAIL_NUM", "DISTANCE", "FL_DATE"])
def test_model_accepts_nulls_in_text_float_and_date_columns(tiny_model, field):
    assert list(tiny_model.predict(frame_of(**{field: None}))) == ["on_time"]


@pytest.mark.parametrize("field", ["OP_CARRIER_FL_NUM", "CRS_DEP_TIME", "CRS_ARR_TIME"])
def test_model_rejects_nulls_in_integer_columns(tiny_model, field):
    with pytest.raises(MlflowException) as error:
        tiny_model.predict(frame_of(**{field: None}))

    assert error.value.error_code == "INVALID_PARAMETER_VALUE"
    assert f"column {field}" in str(error.value)


def test_model_rejects_a_missing_column(tiny_model):
    with pytest.raises(MlflowException) as error:
        tiny_model.predict(frame_of().drop(columns=["DISTANCE"]))

    assert error.value.error_code == "INVALID_PARAMETER_VALUE"
    assert "missing inputs ['DISTANCE']" in str(error.value)


def test_model_ignores_extra_columns(tiny_model):
    assert list(tiny_model.predict(frame_of().assign(DEP_TIME=905))) == ["on_time"]


def test_classify_works_end_to_end_with_the_tiny_model(client, tiny_model):
    app.state.model = tiny_model

    response = client.post("/classify", json={"instances": [COMPLETE_ROW, COMPLETE_ROW]})

    assert response.status_code == 200
    assert response.json() == {"classifications": ["on_time", "on_time"]}


def test_classify_answers_422_with_the_reason_for_a_null_integer(client, tiny_model):
    app.state.model = tiny_model

    flight = {**COMPLETE_ROW, "CRS_DEP_TIME": None}

    response = client.post("/classify", json={"instances": [flight]})

    assert response.status_code == 422
    assert response.json()["detail"].startswith("Incompatible input types for column CRS_DEP_TIME")
