"""Governance guardrails for a regulated, data-intensive health setting.

Four concerns, each small and testable:
  * PII redaction      — scrub emails / phones / long digit strings before text ever
                         reaches an external LLM.
  * medical-advice     — detect requests for diagnosis/treatment and steer to a safe,
                         non-diagnostic framing.
  * citation check     — ensure a generated answer references the dates it drew from;
                         uncited answers are rejected.
  * audit log          — append-only record of every query, its context, and provider.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .config import AUDIT_LOG_PATH

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"\b(?:\+?\d[\d\-\s().]{7,}\d)\b")
_LONG_DIGITS = re.compile(r"\b\d{6,}\b")           # ids, MRNs, long account numbers
_DATE_CITATION = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")

_MEDICAL_ADVICE = re.compile(
    r"\b(diagnos\w+|should i take|what medication|prescri\w+|is this (a )?(disease|condition)|"
    r"do i have|treat(ment)? for|cure|injured|see a doctor because)\b",
    re.IGNORECASE,
)

DISCLAIMER = (
    "Note: this is an engineering demonstration over wearable wellness data and is "
    "not medical, diagnostic, or training advice."
)


def redact_pii(text: str) -> str:
    # Order matters: redact emails first, then long standalone IDs, then phone-like
    # runs. Doing IDs before phones stops the phone pattern from swallowing a bare
    # numeric id (e.g. "id 998877665") that should be tagged as an ID.
    text = _EMAIL.sub("[REDACTED_EMAIL]", text)
    text = _LONG_DIGITS.sub("[REDACTED_ID]", text)
    text = _PHONE.sub("[REDACTED_PHONE]", text)
    return text


def is_medical_advice_request(question: str) -> bool:
    return bool(_MEDICAL_ADVICE.search(question))


def enforce_citation(answer: str, allowed_dates: set[str]) -> tuple[bool, str]:
    """An answer is valid only if it cites at least one date from the retrieved set.

    Returns (ok, message). This is the anti-hallucination backstop: if the model
    produced claims without grounding in the dates we actually retrieved, we reject it.
    """
    cited = set(_DATE_CITATION.findall(answer))
    grounded = cited & allowed_dates
    if not grounded:
        return False, "Answer rejected: it did not cite any retrieved dated record."
    unknown = cited - allowed_dates
    if unknown:
        return False, (
            "Answer rejected: it cited dates not present in the retrieved context "
            f"({', '.join(sorted(unknown))})."
        )
    return True, f"Grounded in {len(grounded)} retrieved record(s)."


def audit(event: dict, path: Path = AUDIT_LOG_PATH) -> None:
    event = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **event}
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, default=str) + "\n")
