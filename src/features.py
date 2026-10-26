"""Sports-science feature derivation: raw numbers -> interpreted signal.

This is where domain knowledge lives. The retrieval layer and the LLM both lean on
these derived fields so answers are grounded in pre-computed, auditable signal rather
than on-the-fly arithmetic by a language model.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import SportsScienceThresholds
from .schema import DailyMetric


@dataclass
class DailyFeature:
    athlete_id: str
    date: str
    hrv_state: str
    hrv_vs_baseline: float | None
    sleep_debt_hours: float | None
    acwr_band: str
    readiness_flag: str
    is_rest_day: int


def classify_acwr(acwr: float | None, t: SportsScienceThresholds) -> str:
    """Map the acute:chronic workload ratio onto an injury-risk band."""
    if acwr is None:
        return "unknown"
    if acwr < t.acwr_undertrained_below:
        return "detraining"
    if acwr <= t.acwr_optimal_high:
        return "optimal"
    if acwr <= t.acwr_high_risk_above:
        return "caution"
    return "high_risk"


def classify_hrv(metric: DailyMetric) -> tuple[str, float | None]:
    """Judge HRV against the athlete's own rolling baseline band.

    Returns (state, signed distance from the baseline midpoint in ms). Below the
    baseline-low band reads as suppressed recovery; above the high band as elevated.
    """
    if metric.hrv_ms is None or metric.hrv_baseline_low is None or metric.hrv_baseline_high is None:
        return "unknown", None
    low, high = metric.hrv_baseline_low, metric.hrv_baseline_high
    mid = (low + high) / 2.0
    distance = round(metric.hrv_ms - mid, 1)
    if metric.hrv_ms < low:
        return "suppressed", distance
    if metric.hrv_ms > high:
        return "elevated", distance
    return "normal", distance


def compute_sleep_debt(
    recent_sleep_hours: list[float | None], t: SportsScienceThresholds
) -> float | None:
    """Cumulative sleep debt vs target across a trailing window (positive = debt)."""
    valid = [h for h in recent_sleep_hours if h is not None]
    if not valid:
        return None
    debt = sum(t.sleep_target_hours - h for h in valid)
    return round(debt, 1)


def readiness(hrv_state: str, acwr_band: str, sleep_score: int | None,
              t: SportsScienceThresholds) -> str:
    """Combine recovery + load + sleep into a simple ok/monitor/rest flag."""
    poor_sleep = sleep_score is not None and sleep_score < t.sleep_poor_score_below
    if acwr_band == "high_risk" or (hrv_state == "suppressed" and poor_sleep):
        return "rest"
    if hrv_state == "suppressed" or acwr_band == "caution" or poor_sleep:
        return "monitor"
    return "ok"


def derive_features(
    metric: DailyMetric,
    trailing_sleep_hours: list[float | None],
    t: SportsScienceThresholds,
) -> DailyFeature:
    hrv_state, hrv_dist = classify_hrv(metric)
    acwr_band = classify_acwr(metric.acwr, t)
    sleep_debt = compute_sleep_debt(trailing_sleep_hours, t)
    flag = readiness(hrv_state, acwr_band, metric.sleep_score, t)
    is_rest = 1 if (metric.exercise_min or 0) == 0 else 0
    return DailyFeature(
        athlete_id=metric.athlete_id,
        date=metric.date,
        hrv_state=hrv_state,
        hrv_vs_baseline=hrv_dist,
        sleep_debt_hours=sleep_debt,
        acwr_band=acwr_band,
        readiness_flag=flag,
        is_rest_day=is_rest,
    )
