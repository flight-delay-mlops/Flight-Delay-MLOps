# Offline side of the feature store (the features themselves are defined in api/features.py).
# DVC stage: for every airport, airline and route, and every day they had flights, the delay
# rates of the 30 days *before* that day. Training looks each flight up in Feast, so a
# flight's features never include its own delay or anything after it.
import glob
import os
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import ibis
import pandas as pd
from feast import FeatureStore, RepoConfig

from api.features import KEYS, WINDOW_DAYS, add_route, feast_definitions, feature_names, rates
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


def open_offline_store(folder):
    # Same definitions as the online store, read from the parquet tables with DuckDB (Feast's
    # default engine needs far more memory). A route's table only has the days it had flights
    # and training always asks for one of those, so 1 day of ttl is enough and keeps it fast
    config = RepoConfig(
        project="flight_delay", provider="local", entity_key_serialization_version=3,
        registry=os.path.join(folder, "registry.db"), offline_store={"type": "duckdb"},
        online_store={"type": "sqlite", "path": os.path.join(folder, "online.db")},
    )
    config.repo_path = Path.cwd()  # the tables' paths are relative to the repository
    store = FeatureStore(config=config)
    for key in KEYS:
        store.apply(feast_definitions(key, ttl=timedelta(days=1)))
    # Feast runs DuckDB through ibis: cap it, so the runner (4 GB) has room for the flights.
    # It writes to disk what doesn't fit
    for setting in ("memory_limit = '256MB'", "threads = 2", "preserve_insertion_order = false"):
        ibis.get_backend().raw_sql(f"SET {setting}")
    return store


def add_offline_features(flights):
    # Training: each flight gets its keys' rates from the 30 days before its date (Feast
    # historical features, a join by key and date that never looks past the date)
    flights = add_route(flights)
    flights["FL_DATE"] = pd.to_datetime(flights.FL_DATE).astype("datetime64[ns]")
    with TemporaryDirectory() as folder:  # the definitions are applied again every time
        store = open_offline_store(folder)
        for key in KEYS:
            # Every flight of a key on a day has the same rates: look each pair up once,
            # one month at a time to keep memory low
            pairs = flights[[key, "FL_DATE"]].drop_duplicates()
            months = [historical_rates(store, key, month)
                      for _, month in pairs.groupby(pairs.FL_DATE.dt.to_period("M"))]
            table = pd.concat(months, ignore_index=True)
            flights = flights.merge(table, on=[key, "FL_DATE"], how="left")
    return flights


def historical_rates(store, key, pairs):
    entities = pd.DataFrame({key: pairs[key].astype(str),
                             "event_timestamp": pairs.FL_DATE.dt.tz_localize("UTC")})
    found = store.get_historical_features(entity_df=entities, features=feature_names(key)).to_df()
    days = pd.to_datetime(found.pop("event_timestamp"), utc=True).dt.tz_localize(None)
    # Same key type as the flights', which is also far smaller than Python strings
    return found.assign(FL_DATE=days.astype("datetime64[ns]"),
                        **{key: found[key].astype(pairs[key].dtype)})


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
