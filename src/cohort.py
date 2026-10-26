"""Synthetic cohort generator.

Produces additional athletes with distinct, deterministic physiological profiles so
the storage layer, filters, and dashboard prove out beyond a single person. No
external randomness library is used (stdlib `random` seeded) so output is reproducible.

Each athlete gets a 'persona' that shapes their signal:
  * athlete-b 'heavy-block'  — a hard training block that pushes ACWR into caution.
  * athlete-c 'well-recovered' — balanced load, strong sleep, HRV near baseline.
"""
from __future__ import annotations

import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path

from .config import DATA_DIR

_HEADER = [
    "date", "resting_hr", "hrv_ms", "hrv_baseline_low", "hrv_baseline_high",
    "garmin_stress", "sleep_hours", "sleep_score", "exercise_kcal", "exercise_min",
    "sports", "acute_load", "chronic_load", "acwr", "skin_temp_dev_c",
]

_SPORTS = [
    "running", "running;strength_training", "indoor_cycling",
    "indoor_cycling;strength_training", "cycling", "running;track_running",
    "strength_training", "",
]


def _persona_params(persona: str) -> dict:
    if persona == "heavy-block":
        return {"base_rhr": 48, "base_hrv": 70, "hrv_spread": 10,
                "acute_center": 700, "chronic": 520, "sleep_mu": 6.6, "sleep_sd": 1.2}
    if persona == "well-recovered":
        return {"base_rhr": 42, "base_hrv": 95, "hrv_spread": 8,
                "acute_center": 450, "chronic": 470, "sleep_mu": 7.8, "sleep_sd": 0.7}
    return {"base_rhr": 45, "base_hrv": 85, "hrv_spread": 9,
            "acute_center": 500, "chronic": 500, "sleep_mu": 7.2, "sleep_sd": 1.0}


def generate_athlete(athlete_id: str, persona: str, start: date, days: int,
                     seed: int) -> Path:
    rng = random.Random(seed)
    p = _persona_params(persona)
    rows: list[list] = []

    baseline_low, baseline_high = p["base_hrv"] - 15, p["base_hrv"] + 20
    for i in range(days):
        d = start + timedelta(days=i)
        is_rest = rng.random() < 0.25

        # training load oscillates with a weekly rhythm plus noise
        wave = math.sin(i / 7.0 * math.pi)
        acute = max(0, int(p["acute_center"] + wave * 180 + rng.gauss(0, 90)))
        if is_rest:
            acute = int(acute * 0.3)
        chronic = p["chronic"]
        acwr = round(acute / chronic, 1) if chronic else 0.0

        hrv = round(max(25.0, rng.gauss(p["base_hrv"], p["hrv_spread"])), 1)
        rhr = int(max(35, rng.gauss(p["base_rhr"], 2)))
        sleep_h = round(max(0.0, min(10.0, rng.gauss(p["sleep_mu"], p["sleep_sd"]))), 2)
        sleep_score = int(max(20, min(100, 40 + sleep_h * 7 + rng.gauss(0, 6))))
        ex_min = 0 if is_rest else int(max(10, rng.gauss(90, 40)))
        ex_kcal = 0 if is_rest else int(ex_min * rng.uniform(4, 8))
        stress = int(max(10, min(60, rng.gauss(25, 7))))
        sport = "" if is_rest else rng.choice([s for s in _SPORTS if s])
        skin_temp = round(rng.gauss(0, 0.07), 2)

        rows.append([
            d.isoformat(), rhr, hrv, float(baseline_low), float(baseline_high),
            stress, sleep_h, sleep_score, ex_kcal, ex_min, sport,
            acute, chronic, acwr, skin_temp,
        ])

    out = DATA_DIR / f"{athlete_id}_wearable_daily.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_HEADER)
        writer.writerows(rows)
    return out


def generate_default_cohort(start: date | None = None, days: int = 56) -> list[tuple[str, Path]]:
    start = start or date(2026, 8, 25)
    specs = [
        ("athlete-b", "heavy-block", 101),
        ("athlete-c", "well-recovered", 202),
    ]
    created = []
    for athlete_id, persona, seed in specs:
        path = generate_athlete(athlete_id, persona, start, days, seed)
        created.append((athlete_id, path))
    return created
