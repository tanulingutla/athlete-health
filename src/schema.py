"""SQLite schema (the storage layer) and the typed row model.

The storage layer is split in two:
  * `daily_metrics`  — the validated raw signal, one row per athlete per day.
  * `daily_features` — derived, interpreted signal (HRV state, sleep debt, ACWR band).

Keeping raw and derived separate means we can recompute features when the science
improves without re-ingesting source data, and SQL analytics can join the two.
"""
from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS daily_metrics (
    athlete_id          TEXT NOT NULL,
    date                TEXT NOT NULL,              -- ISO YYYY-MM-DD
    resting_hr          INTEGER,
    hrv_ms              REAL,
    hrv_baseline_low    REAL,
    hrv_baseline_high   REAL,
    garmin_stress       INTEGER,
    sleep_hours         REAL,
    sleep_score         INTEGER,
    exercise_kcal       INTEGER,
    exercise_min        INTEGER,
    sports              TEXT,                       -- ';'-separated, may be empty
    acute_load          REAL,
    chronic_load        REAL,
    acwr                REAL,
    skin_temp_dev_c     REAL,
    content_hash        TEXT NOT NULL,              -- for dedupe / change detection
    ingested_at         TEXT NOT NULL,
    PRIMARY KEY (athlete_id, date)
);

CREATE TABLE IF NOT EXISTS daily_features (
    athlete_id          TEXT NOT NULL,
    date                TEXT NOT NULL,
    hrv_state           TEXT,       -- suppressed | normal | elevated
    hrv_vs_baseline     REAL,       -- signed distance from baseline midpoint (ms)
    sleep_debt_hours    REAL,       -- cumulative vs target over trailing window
    acwr_band           TEXT,       -- detraining | optimal | caution | high_risk
    readiness_flag      TEXT,       -- ok | monitor | rest
    is_rest_day         INTEGER,    -- 1 if no exercise logged
    PRIMARY KEY (athlete_id, date),
    FOREIGN KEY (athlete_id, date)
        REFERENCES daily_metrics (athlete_id, date)
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    athlete_id          TEXT NOT NULL,
    source              TEXT NOT NULL,
    rows_seen           INTEGER NOT NULL,
    rows_written        INTEGER NOT NULL,
    rows_skipped_dupe   INTEGER NOT NULL,
    rows_rejected       INTEGER NOT NULL,
    earliest_date       TEXT,
    latest_date         TEXT,
    ran_at              TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_metrics_date ON daily_metrics (date);
CREATE INDEX IF NOT EXISTS idx_features_band ON daily_features (acwr_band);
"""


@dataclass
class DailyMetric:
    """One validated day of wearable signal."""
    athlete_id: str
    date: str
    resting_hr: int | None
    hrv_ms: float | None
    hrv_baseline_low: float | None
    hrv_baseline_high: float | None
    garmin_stress: int | None
    sleep_hours: float | None
    sleep_score: int | None
    exercise_kcal: int | None
    exercise_min: int | None
    sports: str
    acute_load: float | None
    chronic_load: float | None
    acwr: float | None
    skin_temp_dev_c: float | None
    content_hash: str = ""
    ingested_at: str = ""

    def as_row(self) -> dict:
        return asdict(self)


def connect(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Path | str) -> None:
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
