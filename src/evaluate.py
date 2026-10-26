"""Retrieval evaluation harness.

The JD calls out 'iterate on context relevance and accuracy' and 'retrieval
optimization'. This module turns that into something measurable: a small, curated
set of questions, each with the dates a correct retrieval *should* surface, scored
with standard IR metrics.

Metrics (computed per query, then macro-averaged):
  * precision@k  — of the top-k retrieved, how many are relevant.
  * recall@k     — of the relevant dates, how many were retrieved.
  * hit@k        — did at least one relevant date appear in the top-k.
  * MRR          — mean reciprocal rank of the first relevant hit.

The gold set is grounded in the actual data (verified against SQL analytics): the
only caution-zone load day is 2026-09-03; suppressed-HRV days cluster in late Sept /
early Oct; the worst sleep nights are 2026-10-19 and 2026-10-11.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .retriever import Retriever


@dataclass
class EvalCase:
    question: str
    relevant_dates: set[str]
    note: str = ""


# Gold set for athlete-a. Dates come from the ground-truth SQL analytics.
GOLD_SET: list[EvalCase] = [
    EvalCase(
        "When was my training load dangerously high?",
        {"2026-09-03"},
        "Only caution-zone ACWR day (1.4).",
    ),
    EvalCase(
        "Was I overtraining in early September?",
        {"2026-09-03"},
        "Early-Sept high-load day.",
    ),
    EvalCase(
        "Which days was my HRV suppressed and recovery low?",
        {"2026-09-19", "2026-09-25", "2026-09-26", "2026-10-02", "2026-10-06", "2026-10-07"},
        "Suppressed-HRV cluster.",
    ),
    EvalCase(
        "When did I sleep the worst?",
        {"2026-10-19", "2026-10-11", "2026-10-06", "2026-09-26", "2026-09-16"},
        "Five lowest sleep-score nights.",
    ),
]


@dataclass
class QueryScore:
    question: str
    precision_at_k: float
    recall_at_k: float
    hit_at_k: float
    reciprocal_rank: float
    retrieved_dates: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    per_query: list[QueryScore]
    k: int

    @property
    def mean_precision(self) -> float:
        return _mean(s.precision_at_k for s in self.per_query)

    @property
    def mean_recall(self) -> float:
        return _mean(s.recall_at_k for s in self.per_query)

    @property
    def hit_rate(self) -> float:
        return _mean(s.hit_at_k for s in self.per_query)

    @property
    def mrr(self) -> float:
        return _mean(s.reciprocal_rank for s in self.per_query)


def _mean(values) -> float:
    vals = list(values)
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _score_query(case: EvalCase, retrieved_dates: list[str], k: int) -> QueryScore:
    top = retrieved_dates[:k]
    relevant_in_top = [d for d in top if d in case.relevant_dates]
    precision = len(relevant_in_top) / len(top) if top else 0.0
    recall = (len(set(top) & case.relevant_dates) / len(case.relevant_dates)
              if case.relevant_dates else 0.0)
    hit = 1.0 if relevant_in_top else 0.0
    rr = 0.0
    for rank, d in enumerate(top, start=1):
        if d in case.relevant_dates:
            rr = 1.0 / rank
            break
    return QueryScore(
        question=case.question,
        precision_at_k=round(precision, 4),
        recall_at_k=round(recall, 4),
        hit_at_k=hit,
        reciprocal_rank=round(rr, 4),
        retrieved_dates=top,
    )


def evaluate(retriever: Retriever | None = None, k: int = 6,
             gold: list[EvalCase] | None = None) -> EvalReport:
    retriever = retriever or Retriever()
    gold = gold or GOLD_SET
    scores = []
    for case in gold:
        results = retriever.retrieve(case.question, top_k=k)
        dates = [r["metadata"].get("date") for r in results]
        scores.append(_score_query(case, dates, k))
    return EvalReport(per_query=scores, k=k)
