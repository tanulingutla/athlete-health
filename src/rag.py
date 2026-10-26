"""RAG orchestration: tie retrieval, guardrails, and generation together.

Flow:
  question
    -> medical-advice guard (reframe / disclaim)
    -> PII redaction
    -> retrieve dated daily narratives (hybrid)
    -> build a cited prompt
    -> LLM (or offline answerer)
    -> citation enforcement (anti-hallucination)
    -> audit log
    -> answer + sources
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import guardrails
from .llm import LLMClient
from .retriever import Retriever

SYSTEM_PROMPT = (
    "You are an analyst that summarises an athlete's wearable training and recovery "
    "data. Answer ONLY from the dated records in the CONTEXT. Cite the specific dates "
    "(YYYY-MM-DD) you rely on. If the context does not support an answer, say so. "
    "Never give medical, diagnostic, or treatment advice. Be concise and factual."
)


@dataclass
class RagResult:
    question: str
    answer: str
    provider: str
    sources: list[dict] = field(default_factory=list)
    citation_ok: bool = True
    citation_note: str = ""
    guard_triggered: bool = False


def _build_user_prompt(question: str, results: list[dict]) -> str:
    context_lines = []
    for r in results:
        date = r["metadata"].get("date", "unknown")
        context_lines.append(f"[{date}] {r['document']}")
    context = "\n".join(context_lines) if context_lines else "(no records retrieved)"
    return f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer with cited dates."


class RagPipeline:
    def __init__(self, retriever: Retriever | None = None, provider: str | None = None):
        self.retriever = retriever or Retriever()
        self.llm = LLMClient(provider=provider)

    def ask(self, question: str, top_k: int | None = None,
            athlete_id: str | None = None) -> RagResult:
        guard_triggered = guardrails.is_medical_advice_request(question)

        safe_question = guardrails.redact_pii(question)
        results = self.retriever.retrieve(safe_question, top_k=top_k,
                                          athlete_id=athlete_id)
        allowed_dates = {r["metadata"].get("date") for r in results if r["metadata"].get("date")}

        user_prompt = _build_user_prompt(safe_question, results)

        if guard_triggered:
            # Do not attempt a medical answer; return a safe, grounded, non-diagnostic note.
            dates = ", ".join(sorted(allowed_dates)) or "the retrieved period"
            answer = (
                "I can't provide medical or diagnostic advice. I can only describe "
                f"patterns in the recorded data ({dates}). For anything health-related, "
                "please consult a qualified professional.\n\n" + guardrails.DISCLAIMER
            )
            citation_ok, note = (True, "Guarded response; no claims made.")
        else:
            answer = self.llm.generate(SYSTEM_PROMPT, user_prompt)
            if guardrails.DISCLAIMER.split(".")[0] not in answer and "not medical" not in answer.lower():
                answer = f"{answer}\n\n{guardrails.DISCLAIMER}"
            citation_ok, note = guardrails.enforce_citation(answer, allowed_dates)

        sources = [
            {"date": r["metadata"].get("date"),
             "acwr_band": r["metadata"].get("acwr_band"),
             "readiness_flag": r["metadata"].get("readiness_flag"),
             "score": round(r["score"], 4)}
            for r in results
        ]

        guardrails.audit({
            "question": question,
            "redacted_question": safe_question,
            "provider": self.llm.provider,
            "retrieved_dates": sorted(allowed_dates),
            "citation_ok": citation_ok,
            "guard_triggered": guard_triggered,
        })

        return RagResult(
            question=question,
            answer=answer,
            provider=self.llm.provider,
            sources=sources,
            citation_ok=citation_ok,
            citation_note=note,
            guard_triggered=guard_triggered,
        )
