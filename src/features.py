# Offline side of the feature store (the features themselves are defined in api/features.py).
# DVC stage: for every airport, airline and route, and every day they had flights, the delay
# rates of the 30 days *before* that day. Training looks each flight up by key and date, so a
# flight's features never include its own delay or anything after it.
import glob
import os

import pandas as pd

from api.features import KEYS, WINDOW_DAYS, add_route, rates
from src.prepare import filter_flights

OUT_DIR = "data/features"


def daily_counts(path):
    # Per key and day: number of flights, total arrival delay and how many were 15+ min late
    flights = pd.read_csv(path, engine="pyarrow", usecols=[
        "FL_DATE", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "ARR_DELAY", "CANCELLED", "DIVERTED"])
    flights = add_route(filter_flights(flights))
    flights["FL_DATE"] = pd.to_datetime(flights.FL_DATE)
    flights["delayed"] = flights.ARR_DELAY >= 15
    return {key: flights.groupby([key, "FL_DATE"]).agg(
                flights=("delayed", "size"), delay=("ARR_DELAY", "sum"), delayed=("delayed", "sum"))
            for key in KEYS}


def add_offline_features(flights):
    # Training: each flight gets its keys' rates from the 30 days before its date
    flights = add_route(flights)
    flights["FL_DATE"] = pd.to_datetime(flights.FL_DATE)
    for key in KEYS:
        table = pd.read_parquet(f"{OUT_DIR}/{key}.parquet", dtype_backend="pyarrow")
        table["FL_DATE"] = table.FL_DATE.astype("datetime64[ns]")  # same type as the flights'
        # Only the days these flights need (the route table alone has 3.5M rows)
        table = table[table.FL_DATE.between(flights.FL_DATE.min(), flights.FL_DATE.max())]
        flights = flights.merge(table, on=[key, "FL_DATE"], how="left")
    return flights


def main():
    # One monthly file at a time: memory stays low on the 4 GB runner
    parts = [daily_counts(path) for path in sorted(glob.glob("data/raw/*.csv.gz"))]
    os.makedirs(OUT_DIR, exist_ok=True)
    for key in KEYS:
        counts = pd.concat(part[key] for part in parts).reset_index().sort_values("FL_DATE")
        # closed="left": the window ends just before the day, leaving the day itself out
        window = (counts.set_index("FL_DATE").groupby(key)[["flights", "delay", "delayed"]]
                  .rolling(f"{WINDOW_DAYS}D", closed="left").sum())
        table = pd.concat([window.reset_index()[[key, "FL_DATE"]],
                           rates(window, key).reset_index(drop=True)], axis=1)
        table.to_parquet(f"{OUT_DIR}/{key}.parquet", index=False)
        print(f"{key}: {len(table):,} rows")


if __name__ == "__main__":
    main()
