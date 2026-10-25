from datetime import date

import pytest
from pydantic import ValidationError

from api.schemas import ClassifyRequest, ClassifyResponse, Flight

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


def test_flight_accepts_a_complete_row():
    flight = Flight(**COMPLETE_ROW)

    assert flight.FL_DATE == date(2023, 7, 2)
    assert flight.CRS_DEP_TIME == 900
    assert flight.DISTANCE == 528.0
    assert flight.model_dump(mode="json") == COMPLETE_ROW


def test_missing_keys_count_as_null():
    flight = Flight(ORIGIN="DTW")

    assert flight.ORIGIN == "DTW"
    assert all(value is None for name, value in flight if name != "ORIGIN")


def test_null_values_are_accepted():
    flight = Flight(**{name: None for name in Flight.model_fields})

    assert all(value is None for _, value in flight)


def test_unknown_fields_are_ignored():
    flight = Flight(**COMPLETE_ROW, DEP_TIME=905, ARR_DELAY=12.0)

    assert "DEP_TIME" not in flight.model_dump()
    assert "ARR_DELAY" not in flight.model_dump()


@pytest.mark.parametrize(
    "field, raw, expected",
    [
        ("CRS_DEP_TIME", "900", 900),
        ("CRS_ARR_TIME", 1042.0, 1042),
        ("DISTANCE", "528.5", 528.5),
        ("FL_DATE", "2023-07-02", date(2023, 7, 2)),
    ],
)
def test_values_are_converted_to_the_field_type(field, raw, expected):
    assert getattr(Flight(**{field: raw}), field) == expected


@pytest.mark.parametrize(
    "field, raw",
    [
        ("CRS_DEP_TIME", "abc"),
        ("OP_CARRIER_FL_NUM", 12.5),
        ("DISTANCE", "far"),
        ("FL_DATE", "not-a-date"),
        ("FL_DATE", "02/07/2023"),
    ],
)
def test_invalid_values_are_rejected(field, raw):
    with pytest.raises(ValidationError):
        Flight(**{field: raw})


def test_request_keeps_the_flights_in_order():
    request = ClassifyRequest(instances=[{"ORIGIN": "DTW"}, COMPLETE_ROW, {"ORIGIN": "JFK"}])

    assert [flight.ORIGIN for flight in request.instances] == ["DTW", "DTW", "JFK"]
    assert all(isinstance(flight, Flight) for flight in request.instances)


def test_request_treats_an_empty_object_as_a_flight_with_every_field_null():
    request = ClassifyRequest(instances=[{}])

    assert all(value is None for _, value in request.instances[0])


@pytest.mark.parametrize(
    "payload",
    [{}, {"instances": None}, {"instances": []}, {"instances": "DTW"}, {"instances": [5]}],
    ids=["missing", "null", "empty", "not-a-list", "element-not-an-object"],
)
def test_request_rejects_a_malformed_instances_array(payload):
    with pytest.raises(ValidationError):
        ClassifyRequest.model_validate(payload)


def test_to_dataframe_has_a_row_per_flight_in_order_and_a_column_per_field():
    request = ClassifyRequest(instances=[{"ORIGIN": "AAA"}, {"ORIGIN": "BBB"}, {"ORIGIN": "CCC"}])

    frame = request.to_dataframe()

    assert list(frame.ORIGIN) == ["AAA", "BBB", "CCC"]
    assert list(frame.columns) == list(Flight.model_fields)


def test_to_dataframe_keeps_nulls_for_the_model_to_judge():
    frame = ClassifyRequest(instances=[{**COMPLETE_ROW, "TAIL_NUM": None}]).to_dataframe()

    assert frame.TAIL_NUM.isna().all()
    assert frame.DISTANCE.iloc[0] == 528.0


def test_request_error_names_the_instance_and_the_field():
    with pytest.raises(ValidationError) as error:
        ClassifyRequest.model_validate({"instances": [{}, {"DISTANCE": "far"}]})

    assert error.value.errors()[0]["loc"] == ("instances", 1, "DISTANCE")


def test_response_accepts_the_three_labels_in_order():
    labels = ["moderate_delay", "on_time", "severe_delay"]

    assert ClassifyResponse(classifications=labels).model_dump() == {"classifications": labels}


@pytest.mark.parametrize("label", ["late", "ON_TIME", ""])
def test_response_rejects_unknown_labels(label):
    with pytest.raises(ValidationError):
        ClassifyResponse(classifications=["on_time", label])
