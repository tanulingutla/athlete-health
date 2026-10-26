"""Ingestion: pull daily records from a source, validate, de-duplicate, derive
features, and write to the storage layer. Tracks freshness and reports gaps.

Two source adapters are provided:
  * CsvSource         — reads a wearable CSV export (the data we have today).
  * LiveApiSource     — a stub for a real wearable API (Garmin/Apple Health), showing
                        the retry/backoff + rate-limit + pagination handling an AI
                        engineer owns when the connector is not pre-built.

Both yield raw dict rows; the shared pipeline does validation/dedupe/persist so the
hard part is written once and reused.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Protocol

from .config import THRESHOLDS
from .features import derive_features
from .schema import DailyMetric, connect, init_db

log = logging.getLogger("ingest")

# Columns we expect from a wearable daily export.
_NUMERIC_INT = {"resting_hr", "garmin_stress", "sleep_score", "exercise_kcal", "exercise_min"}
_NUMERIC_FLOAT = {
    "hrv_ms", "hrv_baseline_low", "hrv_baseline_high", "sleep_hours",
    "acute_load", "chronic_load", "acwr", "skin_temp_dev_c",
}


@dataclass
class IngestReport:
    rows_seen: int = 0
    rows_written: int = 0
    rows_skipped_dupe: int = 0
    rows_rejected: int = 0
    rejected_detail: list[str] = None  # type: ignore[assignment]
    earliest_date: str | None = None
    latest_date: str | None = None
    gaps: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.rejected_detail is None:
            self.rejected_detail = []
        if self.gaps is None:
            self.gaps = []


# --------------------------------------------------------------------------- sources
class Source(Protocol):
    name: str

    def rows(self) -> Iterator[dict]:  # pragma: no cover - interface
        ...


class CsvSource:
    """Reads a wearable CSV export into raw string-keyed dict rows."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.name = f"csv:{self.path.name}"

    def rows(self) -> Iterator[dict]:
        with self.path.open("r", encoding="utf-8", newline="") as fh:
            yield from csv.DictReader(fh)


class LiveApiSource:
    """Adapter for a live wearable API (e.g. Garmin Health / Apple HealthKit export).

    This is a demonstration stub: it does NOT hit a real endpoint by default (so the
    project runs offline), but it implements the production-shaped concerns an AI
    engineer handles when a pre-built connector does not exist:

      * paginated day-by-day fetch over a date range,
      * exponential backoff with jitter on transient errors / 429s,
      * a rate-limit sleep between pages,
      * surfacing permanent failures instead of silently dropping data.

    Wire a real `fetch_page` callable to go live; everything else is reusable.
    """

    def __init__(
        self,
        athlete_id: str,
        start: date,
        end: date,
        fetch_page=None,
        max_retries: int = 4,
        base_delay: float = 0.5,
        rate_limit_delay: float = 0.0,
    ):
        self.athlete_id = athlete_id
        self.start = start
        self.end = end
        self.fetch_page = fetch_page or self._simulated_fetch_page
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.rate_limit_delay = rate_limit_delay
        self.name = f"live_api:{athlete_id}"

    def rows(self) -> Iterator[dict]:
        day = self.start
        while day <= self.end:
            payload = self._fetch_with_backoff(day)
            if payload is not None:
                yield payload
            if self.rate_limit_delay:
                time.sleep(self.rate_limit_delay)
            day += timedelta(days=1)

    def _fetch_with_backoff(self, day: date) -> dict | None:
        attempt = 0
        while True:
            try:
                return self.fetch_page(self.athlete_id, day)
            except TransientFetchError as exc:
                attempt += 1
                if attempt > self.max_retries:
                    log.error("giving up on %s after %d retries: %s", day, attempt - 1, exc)
                    raise
                delay = self.base_delay * (2 ** (attempt - 1))
                # deterministic 'jitter' avoids importing random; good enough for backoff
                jitter = (hash((day.toordinal(), attempt)) % 100) / 1000.0
                sleep_for = delay + jitter
                log.warning("transient error on %s (attempt %d), backing off %.2fs: %s",
                            day, attempt, sleep_for, exc)
                time.sleep(sleep_for)
            except PermanentFetchError as exc:
                log.error("permanent failure on %s, skipping: %s", day, exc)
                return None

    @staticmethod
    def _simulated_fetch_page(athlete_id: str, day: date) -> dict | None:
        # Offline stub: a real implementation issues an HTTP GET here.
        return None


class TransientFetchError(Exception):
    """Retryable error (timeout, 429, 5xx)."""


class PermanentFetchError(Exception):
    """Non-retryable error (404, auth, malformed)."""


# ------------------------------------------------------------------- validation
def _parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _content_hash(row: dict) -> str:
    """Stable hash of the signal fields, used for dedupe and change detection."""
    keyed = {k: row.get(k) for k in sorted(row) if k not in {"content_hash", "ingested_at"}}
    blob = json.dumps(keyed, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def validate_row(raw: dict, athlete_id: str) -> tuple[DailyMetric | None, str | None]:
    """Coerce + range-check a raw row. Returns (metric, error)."""
    raw = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items()}
    date_str = raw.get("date")
    if not date_str:
        return None, "missing date"
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None, f"bad date format: {date_str!r}"

    try:
        values = {k: _parse_int(raw.get(k)) for k in _NUMERIC_INT}
        values.update({k: _parse_float(raw.get(k)) for k in _NUMERIC_FLOAT})
    except (ValueError, TypeError) as exc:
        return None, f"non-numeric field on {date_str}: {exc}"

    # Light physiological range checks — reject impossible values rather than embed them.
    rhr = values["resting_hr"]
    if rhr is not None and not (20 <= rhr <= 120):
        return None, f"resting_hr out of range on {date_str}: {rhr}"
    sh = values["sleep_hours"]
    if sh is not None and not (0 <= sh <= 24):
        return None, f"sleep_hours out of range on {date_str}: {sh}"
    hrv = values["hrv_ms"]
    if hrv is not None and not (0 < hrv <= 300):
        return None, f"hrv_ms out of range on {date_str}: {hrv}"

    metric = DailyMetric(
        athlete_id=athlete_id,
        date=date_str,
        resting_hr=values["resting_hr"],
        hrv_ms=values["hrv_ms"],
        hrv_baseline_low=values["hrv_baseline_low"],
        hrv_baseline_high=values["hrv_baseline_high"],
        garmin_stress=values["garmin_stress"],
        sleep_hours=values["sleep_hours"],
        sleep_score=values["sleep_score"],
        exercise_kcal=values["exercise_kcal"],
        exercise_min=values["exercise_min"],
        sports=raw.get("sports") or "",
        acute_load=values["acute_load"],
        chronic_load=values["chronic_load"],
        acwr=values["acwr"],
        skin_temp_dev_c=values["skin_temp_dev_c"],
    )
    metric.content_hash = _content_hash(metric.as_row())
    metric.ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return metric, None


# ---------------------------------------------------------------------- pipeline
def _existing_hashes(conn: sqlite3.Connection, athlete_id: str) -> dict[str, str]:
    cur = conn.execute(
        "SELECT date, content_hash FROM daily_metrics WHERE athlete_id = ?", (athlete_id,)
    )
    return {r["date"]: r["content_hash"] for r in cur.fetchall()}


def _detect_gaps(dates: list[str]) -> list[str]:
    """Report missing calendar days between the earliest and latest ingested date."""
    if len(dates) < 2:
        return []
    parsed = sorted(datetime.strptime(d, "%Y-%m-%d").date() for d in dates)
    present = set(parsed)
    gaps = []
    cur = parsed[0]
    while cur <= parsed[-1]:
        if cur not in present:
            gaps.append(cur.isoformat())
        cur += timedelta(days=1)
    return gaps


def ingest(
    source: Source,
    athlete_id: str,
    db_path: Path | str,
) -> IngestReport:
    """Run a source through validation + dedupe + feature derivation into SQLite."""
    init_db(db_path)
    conn = connect(db_path)
    report = IngestReport()
    try:
        existing = _existing_hashes(conn, athlete_id)
        # buffer sleep-hours per date so feature derivation can use a trailing window
        validated: list[DailyMetric] = []
        for raw in source.rows():
            report.rows_seen += 1
            metric, err = validate_row(raw, athlete_id)
            if err:
                report.rows_rejected += 1
                report.rejected_detail.append(err)
                continue
            # dedupe: same date + identical content hash => already have it
            if existing.get(metric.date) == metric.content_hash:
                report.rows_skipped_dupe += 1
                continue
            validated.append(metric)

        validated.sort(key=lambda m: m.date)
        sleep_series = {m.date: m.sleep_hours for m in validated}
        ordered_dates = [m.date for m in validated]

        for i, metric in enumerate(validated):
            # trailing 7-day sleep window (inclusive) for sleep-debt feature
            window_dates = ordered_dates[max(0, i - 6): i + 1]
            trailing = [sleep_series[d] for d in window_dates]
            feature = derive_features(metric, trailing, THRESHOLDS)
            _upsert_metric(conn, metric)
            _upsert_feature(conn, feature)
            report.rows_written += 1

        all_dates = [r["date"] for r in conn.execute(
            "SELECT date FROM daily_metrics WHERE athlete_id = ? ORDER BY date",
            (athlete_id,)).fetchall()]
        if all_dates:
            report.earliest_date, report.latest_date = all_dates[0], all_dates[-1]
            report.gaps = _detect_gaps(all_dates)

        _record_run(conn, athlete_id, source.name, report)
        conn.commit()
    finally:
        conn.close()
    return report


def _upsert_metric(conn: sqlite3.Connection, m: DailyMetric) -> None:
    row = m.as_row()
    cols = ", ".join(row.keys())
    placeholders = ", ".join("?" for _ in row)
    updates = ", ".join(f"{k}=excluded.{k}" for k in row if k not in {"athlete_id", "date"})
    conn.execute(
        f"INSERT INTO daily_metrics ({cols}) VALUES ({placeholders}) "
        f"ON CONFLICT(athlete_id, date) DO UPDATE SET {updates}",
        list(row.values()),
    )


def _upsert_feature(conn: sqlite3.Connection, f) -> None:
    conn.execute(
        """INSERT INTO daily_features
             (athlete_id, date, hrv_state, hrv_vs_baseline, sleep_debt_hours,
              acwr_band, readiness_flag, is_rest_day)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(athlete_id, date) DO UPDATE SET
             hrv_state=excluded.hrv_state,
             hrv_vs_baseline=excluded.hrv_vs_baseline,
             sleep_debt_hours=excluded.sleep_debt_hours,
             acwr_band=excluded.acwr_band,
             readiness_flag=excluded.readiness_flag,
             is_rest_day=excluded.is_rest_day""",
        (f.athlete_id, f.date, f.hrv_state, f.hrv_vs_baseline, f.sleep_debt_hours,
         f.acwr_band, f.readiness_flag, f.is_rest_day),
    )


def _record_run(conn: sqlite3.Connection, athlete_id: str, source: str,
                r: IngestReport) -> None:
    conn.execute(
        """INSERT INTO ingest_runs
             (athlete_id, source, rows_seen, rows_written, rows_skipped_dupe,
              rows_rejected, earliest_date, latest_date, ran_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (athlete_id, source, r.rows_seen, r.rows_written, r.rows_skipped_dupe,
         r.rows_rejected, r.earliest_date, r.latest_date,
         datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )
