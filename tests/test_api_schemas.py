from datetime import date

import pytest
from pydantic import ValidationError

from api.schemas import Flight

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
