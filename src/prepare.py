# Turns the raw BTS files into clean train/validation sets.
import glob
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# Columns known before the scheduled departure, the only ones the model may use
FEATURES = [
    "FL_DATE", "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "TAIL_NUM",
    "ORIGIN", "ORIGIN_STATE_ABR", "DEST", "DEST_STATE_ABR",
    "CRS_DEP_TIME", "DEP_TIME_BLK", "CRS_ARR_TIME", "ARR_TIME_BLK",
    "CRS_ELAPSED_TIME", "DISTANCE",
]

LABELS = ["on_time", "moderate_delay", "severe_delay"]

# Outcome columns, used only to filter and label flights, never as features
OUTCOMES = ["CANCELLED", "DIVERTED", "ARR_DELAY"]


def raw_files(raw_dir):
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv.gz")))
    if not files:
        raise FileNotFoundError(f"no .csv.gz files in {raw_dir}")
    return files


def read_raw(path):
    # Only the columns we use, with text kept in compact Arrow form instead of one
    # Python object per value: both keep memory low enough for the 4 GB runner VM
    return pd.read_csv(path, engine="pyarrow", usecols=FEATURES + OUTCOMES,
                       dtype_backend="pyarrow")


def filter_flights(df):
    # Keep only flights that arrived: not cancelled, not diverted, with an arrival delay
    return df[(df.CANCELLED == 0) & (df.DIVERTED == 0) & df.ARR_DELAY.notna()]


def label_delays(arr_delay):
    # Label from the arrival delay: <15 min, 15-59 min, >=60 min
    return pd.cut(
        arr_delay,
        bins=[-float("inf"), 15, 60, float("inf")],
        labels=LABELS,
        right=False,
    )


def split_by_date(df, val_start):
    # Time-based split: train on older flights, validate on the most recent ones
    is_val = pd.to_datetime(df.FL_DATE) >= val_start
    return df[~is_val], df[is_val]


def clean(df):
    df = filter_flights(df)
    return df.assign(label=label_delays(df.ARR_DELAY))[FEATURES + ["label"]]


def prepare(df, val_start):
    return split_by_date(clean(df), val_start)


def main(raw_dir="data/raw", out_dir="data/prepared", params_path="params.yaml"):
    with open(params_path) as f:
        params = yaml.safe_load(f)["prepare"]

    os.makedirs(out_dir, exist_ok=True)
    writers = {}
    counts = {"train": 0, "val": 0}
    try:
        # One monthly file at a time, appended to the outputs: memory stays at about
        # one month of data however many files there are (the runner VM has 4 GB)
        for path in raw_files(raw_dir):
            train, val = prepare(read_raw(path), params["val_start"])
            for name, part in (("train", train), ("val", val)):
                table = pa.Table.from_pandas(part, preserve_index=False)
                if name not in writers:
                    out_path = os.path.join(out_dir, f"{name}.parquet")
                    writers[name] = pq.ParquetWriter(out_path, table.schema)
                writers[name].write_table(table.cast(writers[name].schema))
                counts[name] += len(part)
    finally:
        for writer in writers.values():
            writer.close()

    print(f"train: {counts['train']:,} flights, val: {counts['val']:,} flights")


if __name__ == "__main__":
    main()
