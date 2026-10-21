import pandas as pd
import pytest
import yaml

from src.prepare import FEATURES, filter_flights, label_delays, main, prepare, split_by_date


def make_flights(**overrides):
    # Three valid flights with every raw column prepare.py needs; overrides replace columns
    n = 3
    data = {col: ["x"] * n for col in FEATURES}
    data.update(
        FL_DATE=["2023-06-30", "2023-07-01", "2023-07-02"],
        CRS_DEP_TIME=[900, 1200, 1800],
        DISTANCE=[300.0, 500.0, 800.0],
        CANCELLED=[0.0] * n,
        DIVERTED=[0.0] * n,
        ARR_DELAY=[0.0, 20.0, 90.0],
    )
    data.update(overrides)
    return pd.DataFrame(data)


@pytest.mark.parametrize(
    "delay, label",
    [
        (-30, "on_time"),
        (0, "on_time"),
        (14.9, "on_time"),
        (15, "moderate_delay"),
        (59.9, "moderate_delay"),
        (60, "severe_delay"),
        (600, "severe_delay"),
    ],
)
def test_label_boundaries(delay, label):
    assert label_delays(pd.Series([delay]))[0] == label


def test_filter_drops_cancelled_diverted_and_missing_delay():
    df = make_flights(
        CANCELLED=[1.0, 0.0, 0.0],
        DIVERTED=[0.0, 1.0, 0.0],
        ARR_DELAY=[None, None, 5.0],
    )
    df.loc[0, "ARR_DELAY"] = 10.0  # cancelled but with a delay: still dropped

    kept = filter_flights(df)

    assert len(kept) == 1
    assert kept.iloc[0].ARR_DELAY == 5.0


def test_split_by_date_puts_val_start_in_validation():
    train, val = split_by_date(make_flights(), "2023-07-01")

    assert list(train.FL_DATE) == ["2023-06-30"]
    assert list(val.FL_DATE) == ["2023-07-01", "2023-07-02"]


def test_prepare_keeps_only_pre_departure_features_and_label():
    train, val = prepare(make_flights(), "2023-07-01")

    for part in (train, val):
        assert list(part.columns) == FEATURES + ["label"]
    # Outcome columns would leak the answer
    assert "ARR_DELAY" not in train.columns
    assert list(val.label) == ["moderate_delay", "severe_delay"]


@pytest.fixture
def params_path(tmp_path):
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump({"prepare": {"val_start": "2023-07-01"}}))
    return path


def test_main_writes_train_and_val_parquet(tmp_path, params_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # Real raw files have many more columns, e.g. the actual arrival time: they must be ignored
    flights = make_flights(ARR_TIME=[905, 1230, 1945])
    flights.to_csv(raw_dir / "2023_01.csv.gz", index=False)
    out_dir = tmp_path / "prepared"

    main(raw_dir=raw_dir, out_dir=out_dir, params_path=params_path)

    train = pd.read_parquet(out_dir / "train.parquet")
    val = pd.read_parquet(out_dir / "val.parquet")
    assert len(train) == 1
    assert len(val) == 2
    assert "ARR_TIME" not in train.columns


def test_main_fails_clearly_without_raw_files(tmp_path, params_path):
    with pytest.raises(FileNotFoundError):
        main(raw_dir=tmp_path / "missing", out_dir=tmp_path / "out", params_path=params_path)
