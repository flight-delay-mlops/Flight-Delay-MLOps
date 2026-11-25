"""Feature store: how late flights of each airport, airline and route were in the last 30 days.

The same features exist in two places, defined once here:
  - offline (src/features.py, a DVC stage): for each day, the rates of the 30 days before it,
    from the 2022-2023 data. Training asks Feast for these as they were on each flight's date,
    so a flight never sees its own delay.
  - online (Feast, SQLite files in DATA_DIR/feature_store): the latest rates of each key, read by
    /classify. They are filled from the true outcomes the leaderboard sends to /labels (live mode
    only), so they follow the 2024 flights we are asked about.
"""
import os
import sqlite3
from functools import cache

import pandas as pd
from feast import Entity, FeatureStore, FeatureView, Field, FileSource, RepoConfig, ValueType
from feast.types import Float64

KEYS = ["ORIGIN", "OP_UNIQUE_CARRIER", "ROUTE"]
FEATURES = [f"{key}_{rate}" for key in KEYS for rate in ("avg_delay", "delayed_share")]
WINDOW_DAYS = 30


def add_route(flights):
    # A route is origin + destination, e.g. "JFK-LAX" (compact Arrow strings: millions of rows)
    origin, dest = (flights[column].astype("string[pyarrow]") for column in ("ORIGIN", "DEST"))
    return flights.assign(ROUTE=origin + "-" + dest)


def rates(counts, key):
    # From flight counts to the two features of a key
    return pd.DataFrame({f"{key}_avg_delay": counts.delay / counts.flights,
                         f"{key}_delayed_share": counts.delayed / counts.flights})


def feature_names(key):
    # How Feast calls the features of a key
    return [f"{key.lower()}_delays:{name}" for name in FEATURES if name.startswith(key + "_")]


def feast_definitions(key, ttl=None):
    # Feast's description of one key: who it is (entity) and which features it has (view).
    # ttl: how old the rates may be when looking up the past; serving has no limit
    entity = Entity(name=key.lower(), join_keys=[key], value_type=ValueType.STRING)
    view = FeatureView(
        name=f"{key.lower()}_delays", entities=[entity], ttl=ttl,
        schema=[Field(name=name, dtype=Float64) for name in FEATURES if name.startswith(key + "_")],
        source=FileSource(path=f"data/features/{key}.parquet", timestamp_field="FL_DATE"),
    )
    return [entity, view]


@cache
def open_store(folder):
    # Feast keeps its definitions (registry) and the online values in two SQLite files
    os.makedirs(folder, exist_ok=True)
    store = FeatureStore(config=RepoConfig(
        project="flight_delay", provider="local", entity_key_serialization_version=3,
        registry=os.path.join(folder, "registry.db"),
        online_store={"type": "sqlite", "path": os.path.join(folder, "online.db")},
    ))
    for key in KEYS:
        store.apply(feast_definitions(key))
    return store


def store_folder():
    return os.path.join(os.getenv("DATA_DIR", "."), "feature_store")


def add_online_features(flights):
    # /classify: each flight gets the latest rates of its keys; keys we know nothing about get None
    flights = add_route(flights).reset_index(drop=True)
    found = open_store(store_folder()).get_online_features(
        features=[name for key in KEYS for name in feature_names(key)],
        entity_rows=flights[KEYS].astype(str).to_dict("records"),
    ).to_df()
    return pd.concat([flights, found[FEATURES].astype(float)], axis=1)


def learn_outcomes(flights, outcomes):
    """/labels (live): add the true delays to the daily counts and refresh the online rates."""
    flights = add_route(flights).assign(
        ARR_DELAY=[(outcome or {}).get("ARR_DELAY") for outcome in outcomes],
        day=pd.to_datetime(flights.FL_DATE).dt.strftime("%Y-%m-%d"),
    ).dropna(subset=["ARR_DELAY", "day"])
    flights["delayed"] = flights.ARR_DELAY >= 15
    store = open_store(store_folder())

    with sqlite3.connect(os.path.join(store_folder(), "daily_counts.db")) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS daily_counts (
            key TEXT, value TEXT, day TEXT, flights INTEGER, delay REAL, delayed INTEGER,
            PRIMARY KEY (key, value, day))""")
        for key in KEYS:
            counts = flights.groupby([key, "day"]).agg(
                flights=("delayed", "size"), delay=("ARR_DELAY", "sum"), delayed=("delayed", "sum"))
            db.executemany(
                "INSERT INTO daily_counts VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT DO UPDATE SET "
                "flights = flights + excluded.flights, delay = delay + excluded.delay, "
                "delayed = delayed + excluded.delayed",
                [(key, value, day, int(row.flights), float(row.delay), int(row.delayed))
                 for (value, day), row in counts.iterrows()])

        # The newest values: rates of the last 30 days of outcomes we have, per key
        for key in KEYS:
            recent = pd.read_sql(
                f"SELECT value AS {key}, SUM(flights) AS flights, SUM(delay) AS delay, "
                "SUM(delayed) AS delayed FROM daily_counts WHERE key = ? AND day > "
                f"date((SELECT MAX(day) FROM daily_counts), '-{WINDOW_DAYS} day') GROUP BY value",
                db, params=[key])
            latest = pd.concat([recent[[key]], rates(recent, key)], axis=1)
            latest["FL_DATE"] = pd.Timestamp.now(tz="UTC")
            store.write_to_online_store(f"{key.lower()}_delays", latest)
