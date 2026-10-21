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

params = yaml.safe_load(open("params.yaml"))["prepare"]

files = sorted(glob.glob("data/raw/*.csv.gz"))
df = pd.concat([pd.read_csv(f, engine="pyarrow") for f in files], ignore_index=True)

# Keep only flights that arrived: not cancelled, not diverted, with an arrival delay
df = df[(df.CANCELLED == 0) & (df.DIVERTED == 0) & df.ARR_DELAY.notna()]

# Label from the arrival delay: <15 min, 15-59 min, >=60 min
df["label"] = pd.cut(
    df.ARR_DELAY,
    bins=[-float("inf"), 15, 60, float("inf")],
    labels=["on_time", "moderate_delay", "severe_delay"],
    right=False,
)

# Time-based split: train on older flights, validate on the most recent ones
is_val = pd.to_datetime(df.FL_DATE) >= params["val_start"]
df = df[FEATURES + ["label"]]

os.makedirs("data/prepared", exist_ok=True)
df[~is_val].to_parquet("data/prepared/train.parquet", index=False)
df[is_val].to_parquet("data/prepared/val.parquet", index=False)
print(f"train: {(~is_val).sum():,} flights, val: {is_val.sum():,} flights")
