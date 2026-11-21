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
    "CRS_ELAPSED_TIME", "DISTANCE", "DEP_TIME"
]

LABELS = ["on_time", "moderate_delay", "severe_delay"]

# Outcome columns, used only to filter and label flights, never as features
OUTCOMES = ["CANCELLED", "DIVERTED", "ARR_DELAY"]

NEW_FEATURES = ['ROL_AVG']

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

# Rolling window of the average delay in the last [window_length] time, per airport
def add_rolling_average(df, window_length, lastWindow):
    lastWindowSize = 0
    if lastWindow is not None:
        lastWindowSize = lastWindow.shape[0]
        df = pd.concat([lastWindow, df], ignore_index=True)

    # DEP_TIME is the time, in minutes (hhmm), at which the plane departed, in local time
    # The calculation is the transformation from hhmm to hh*60+mm, without the use of % operator
    # (% operator is not implemented in pyarrow)
    df.index = pd.to_datetime(pd.to_datetime(df.FL_DATE) + pd.to_timedelta(
        df.DEP_TIME - (40*df.DEP_TIME//100), unit='m')
    )
    df = df.sort_index()
    # In case these are no values in window, the rol_avg is the delay (that's why min_periods=1)
    df[NEW_FEATURES[0]] = df.groupby(['ORIGIN'])['ARR_DELAY'].transform(
        lambda x: x.rolling(window=window_length, min_periods=1).mean())

    lastFlightTime = df.index[-1]
    limit = pd.Timestamp(
        year=lastFlightTime.year + lastFlightTime.month//12,
        month=(lastFlightTime.month%12) + 1,
        day=1,
        hour=0,
        minute=0,
        second=0
    ) - pd.Timedelta(window_length)
    newLastWindow = df[df.index > limit]
    return df[lastWindowSize:].reset_index(drop=True), newLastWindow.reset_index(drop=True)

def prepare(df, val_start, window_length, lastWindow):
    df = filter_flights(df)
    df, lastWindow = add_rolling_average(df, window_length, lastWindow)
    df = clean(df)
    train, val = split_by_date(df, val_start)
    return train, val, lastWindow


def main(raw_dir="data/raw", out_dir="data/prepared", params_path="params.yaml"):
    with open(params_path) as f:
        params = yaml.safe_load(f)["prepare"]

    os.makedirs(out_dir, exist_ok=True)
    writers = {}
    counts = {"train": 0, "val": 0}
    lastWindow = None
    try:
        # One monthly file at a time, appended to the outputs: memory stays at about
        # one month of data however many files there are (the runner VM has 4 GB)
        for path in raw_files(raw_dir):
            train, val, lastWindow = prepare(
                read_raw(path),
                params["val_start"],
                params['window_length'],
                lastWindow
            )
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
