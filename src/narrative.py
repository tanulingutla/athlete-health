"""Render a day (metric + derived features) into a natural-language chunk.

This is the heart of the "context layer": instead of embedding raw CSV rows, we embed
a human-readable narrative that already encodes the interpreted signal. Retrieval then
matches on meaning ("overtraining", "poor sleep", "high load") rather than on column
names, and the LLM receives context it can quote directly.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime

from .config import THRESHOLDS


_ACWR_PHRASE = {
    "detraining": "training load was low relative to baseline (possible detraining)",
    "optimal": "training load sat in the optimal range",
    "caution": "training load was elevated (caution zone for injury risk)",
    "high_risk": "training load spiked into the high-risk zone for injury",
    "unknown": "training load ratio was unavailable",
}
_HRV_PHRASE = {
    "suppressed": "HRV was suppressed below the personal baseline band, suggesting under-recovery",
    "normal": "HRV sat within the personal baseline band",
    "elevated": "HRV was elevated above the baseline band",
    "unknown": "HRV reading was unavailable",
}
_READINESS_PHRASE = {
    "ok": "Overall readiness looked good",
    "monitor": "Overall readiness warranted monitoring",
    "rest": "Overall readiness flagged a need to rest",
}


def _weekday(date_str: str) -> str:
    return datetime.strptime(date_str, "%Y-%m-%d").strftime("%A")


def render_day(metric: sqlite3.Row, feature: sqlite3.Row) -> str:
    """Produce one retrievable chunk for a single day."""
    d = metric["date"]
    parts: list[str] = [f"On {d} ({_weekday(d)}):"]

    # Training
    sports = metric["sports"]
    if feature["is_rest_day"]:
        parts.append("It was a rest day with no logged exercise.")
    else:
        sport_txt = sports.replace(";", ", ") if sports else "unspecified activity"
        parts.append(
            f"Trained {metric['exercise_min']} min ({metric['exercise_kcal']} kcal) "
            f"doing {sport_txt}."
        )
    if metric["acwr"] is not None:
        parts.append(
            f"The acute:chronic workload ratio (ACWR) was {metric['acwr']:.1f}; "
            f"{_ACWR_PHRASE.get(feature['acwr_band'], '')}."
        )

    # Recovery
    if metric["hrv_ms"] is not None:
        parts.append(
            f"Morning HRV was {metric['hrv_ms']:.0f} ms against a baseline band of "
            f"{metric['hrv_baseline_low']:.0f}-{metric['hrv_baseline_high']:.0f} ms; "
            f"{_HRV_PHRASE.get(feature['hrv_state'], '')}."
        )
    if metric["resting_hr"] is not None:
        parts.append(f"Resting heart rate was {metric['resting_hr']} bpm.")

    # Sleep
    if metric["sleep_hours"] is not None:
        sleep_txt = f"Slept {metric['sleep_hours']:.1f} h (sleep score {metric['sleep_score']})."
        if feature["sleep_debt_hours"] is not None:
            sleep_txt += (
                f" Trailing 7-day sleep debt stood at {feature['sleep_debt_hours']:.1f} h."
            )
        parts.append(sleep_txt)

    if metric["garmin_stress"] is not None:
        parts.append(f"Daily stress score was {metric['garmin_stress']}.")
    if metric["skin_temp_dev_c"] is not None:
        parts.append(f"Skin temperature deviated {metric['skin_temp_dev_c']:+.2f} C from normal.")

    parts.append(_READINESS_PHRASE.get(feature["readiness_flag"], "") + ".")
    return " ".join(p for p in parts if p.strip())


def render_all(conn: sqlite3.Connection, athlete_id: str) -> list[dict]:
    """Return [{id, date, text, metadata}] for every ingested day of an athlete."""
    rows = conn.execute(
        """SELECT m.*, f.hrv_state, f.hrv_vs_baseline, f.sleep_debt_hours,
                  f.acwr_band, f.readiness_flag, f.is_rest_day
             FROM daily_metrics m
             JOIN daily_features f
               ON m.athlete_id = f.athlete_id AND m.date = f.date
            WHERE m.athlete_id = ?
            ORDER BY m.date""",
        (athlete_id,),
    ).fetchall()

    docs = []
    for r in rows:
        docs.append({
            "id": f"{athlete_id}:{r['date']}",
            "date": r["date"],
            "text": render_day(r, r),
            "metadata": {
                "athlete_id": athlete_id,
                "date": r["date"],
                "acwr_band": r["acwr_band"],
                "hrv_state": r["hrv_state"],
                "readiness_flag": r["readiness_flag"],
                "is_rest_day": r["is_rest_day"],
                "poor_sleep": 1 if (r["sleep_score"] is not None
                                    and r["sleep_score"] < THRESHOLDS.sleep_poor_score_below)
                             else 0,
            },
        })
    return docs
