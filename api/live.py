"""What the leaderboard sends us, kept in SQLite so it survives restarts and redeploys.

Each /classify request is saved with our predictions under its request_id. The true labels
arrive later on /labels and are saved in the same row, which gives our live performance.
On VM2 the database is on a Docker volume (DATA_DIR=/data, see deploy.yml).
"""
import io
import json
import os
import sqlite3

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

LABELS = ["on_time", "moderate_delay", "severe_delay"]


def connect():
    # A new connection per call: FastAPI runs requests in several threads
    db = sqlite3.connect(os.path.join(os.getenv("DATA_DIR", "."), "live.db"))
    db.execute("""CREATE TABLE IF NOT EXISTS requests (
        request_id TEXT PRIMARY KEY, mode TEXT, received_at TEXT DEFAULT CURRENT_TIMESTAMP,
        flights TEXT, predictions TEXT, labels TEXT, outcomes TEXT,
        accuracy REAL, macro_f1 REAL)""")
    return db


def save_request(request_id, mode, flights, predictions):
    with connect() as db:
        db.execute("INSERT OR REPLACE INTO requests (request_id, mode, flights, predictions) "
                   "VALUES (?, ?, ?, ?)",
                   (request_id, mode, flights.to_json(orient="records", date_format="iso"),
                    json.dumps(predictions)))


def save_labels(request_id, labels, outcomes):
    """Store the true labels next to our predictions and score them.

    Returns the request's flights and scores, or None if the id is unknown or was already
    labelled (the leaderboard retries deliveries, and outcomes must not be counted twice).
    """
    with connect() as db:
        row = db.execute("SELECT flights, predictions FROM requests "
                         "WHERE request_id = ? AND labels IS NULL", (request_id,)).fetchone()
        if row is None:
            return None
        flights = pd.read_json(io.StringIO(row[0]), orient="records")
        predictions = json.loads(row[1])
        scores = {
            "accuracy": accuracy_score(labels, predictions),
            # The leaderboard's metric: the three classes count the same
            "macro_f1": f1_score(labels, predictions, labels=LABELS, average="macro",
                                 zero_division=0),
        }
        db.execute("UPDATE requests SET labels = ?, outcomes = ?, accuracy = ?, macro_f1 = ? "
                   "WHERE request_id = ?",
                   (json.dumps(labels), json.dumps(outcomes), scores["accuracy"],
                    scores["macro_f1"], request_id))
    return flights, scores
