# Turns the raw BTS files into clean train/validation sets.
import glob
import os

import pandas as pd
import yaml

# Columns known before the scheduled departure, the only ones the model may use
FEATURES = [
    "FL_DATE", "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "TAIL_NUM",
    "ORIGIN", "ORIGIN_STATE_ABR", "DEST", "DEST_STATE_ABR",
    "CRS_DEP_TIME", "DEP_TIME_BLK", "CRS_ARR_TIME", "ARR_TIME_BLK",
    "CRS_ELAPSED_TIME", "DISTANCE",
]

LABELS = ["on_time", "moderate_delay", "severe_delay"]


def load_raw(raw_dir):
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv.gz")))
    if not files:
        raise FileNotFoundError(f"no .csv.gz files in {raw_dir}")
    return pd.concat([pd.read_csv(f, engine="pyarrow") for f in files], ignore_index=True)


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


def prepare(df, val_start):
    df = filter_flights(df)
    df = df.assign(label=label_delays(df.ARR_DELAY))[FEATURES + ["label"]]
    return split_by_date(df, val_start)


def main(raw_dir="data/raw", out_dir="data/prepared", params_path="params.yaml"):
    with open(params_path) as f:
        params = yaml.safe_load(f)["prepare"]

    train, val = prepare(load_raw(raw_dir), params["val_start"])

    os.makedirs(out_dir, exist_ok=True)
    train.to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
    val.to_parquet(os.path.join(out_dir, "val.parquet"), index=False)
    print(f"train: {len(train):,} flights, val: {len(val):,} flights")


if __name__ == "__main__":
    main()
