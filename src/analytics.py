"""SQL analytics over the storage layer — the 'write queries, not just describe them'
part of the JD. These are plain parameterised SQL statements against SQLite.
"""
from __future__ import annotations

from pathlib import Path

from .schema import connect


def summary(db_path: Path | str, athlete_id: str) -> dict:
    conn = connect(db_path)
    try:
        base = conn.execute(
            """SELECT COUNT(*) AS days,
                      MIN(date) AS first_day,
                      MAX(date) AS last_day,
                      ROUND(AVG(resting_hr), 1) AS avg_rhr,
                      ROUND(AVG(hrv_ms), 1) AS avg_hrv,
                      ROUND(AVG(sleep_hours), 2) AS avg_sleep,
                      ROUND(AVG(sleep_score), 1) AS avg_sleep_score,
                      SUM(exercise_min) AS total_exercise_min
                 FROM daily_metrics WHERE athlete_id = ?""",
            (athlete_id,),
        ).fetchone()

        bands = conn.execute(
            """SELECT acwr_band, COUNT(*) AS n
                 FROM daily_features WHERE athlete_id = ?
                GROUP BY acwr_band ORDER BY n DESC""",
            (athlete_id,),
        ).fetchall()

        high_risk = conn.execute(
            """SELECT m.date, m.acwr, f.readiness_flag
                 FROM daily_metrics m JOIN daily_features f
                   ON m.athlete_id = f.athlete_id AND m.date = f.date
                WHERE m.athlete_id = ? AND f.acwr_band IN ('caution', 'high_risk')
                ORDER BY m.acwr DESC""",
            (athlete_id,),
        ).fetchall()

        suppressed_hrv = conn.execute(
            """SELECT date, hrv_vs_baseline
                 FROM daily_features
                WHERE athlete_id = ? AND hrv_state = 'suppressed'
                ORDER BY date""",
            (athlete_id,),
        ).fetchall()

        worst_sleep = conn.execute(
            """SELECT date, sleep_hours, sleep_score
                 FROM daily_metrics
                WHERE athlete_id = ? AND sleep_score IS NOT NULL
                ORDER BY sleep_score ASC LIMIT 5""",
            (athlete_id,),
        ).fetchall()

        readiness = conn.execute(
            """SELECT readiness_flag, COUNT(*) AS n
                 FROM daily_features WHERE athlete_id = ?
                GROUP BY readiness_flag ORDER BY n DESC""",
            (athlete_id,),
        ).fetchall()

        return {
            "overview": dict(base) if base else {},
            "acwr_band_counts": [dict(r) for r in bands],
            "caution_or_high_risk_days": [dict(r) for r in high_risk],
            "suppressed_hrv_days": [dict(r) for r in suppressed_hrv],
            "worst_sleep_days": [dict(r) for r in worst_sleep],
            "readiness_counts": [dict(r) for r in readiness],
        }
    finally:
        conn.close()
