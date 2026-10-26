"""Streamlit dashboard for the Athlete Health RAG context layer.

Run:  streamlit run app.py

Three panes:
  1. Trends   — ACWR, HRV-vs-baseline, and sleep charted over time from SQLite.
  2. Ask      — a chat box over the RAG pipeline, showing the cited answer + sources.
  3. Retrieval quality — the evaluation harness scores, so a reviewer sees that
                         retrieval accuracy is measured, not assumed.

Offline-first: works with the hashing embedder + JSON store and no API key. If a
real provider / ChromaDB / sentence-transformers is installed, it is used instead.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import streamlit as st

from src.analytics import summary
from src.config import DB_PATH
from src.evaluate import evaluate
from src.rag import RagPipeline
from src.retriever import Retriever
from src.schema import connect


st.set_page_config(page_title="Athlete Health RAG", layout="wide")


# --------------------------------------------------------------------- data access
@st.cache_data(show_spinner=False)
def load_daily(db_path: str, athlete_id: str) -> list[dict]:
    if not Path(db_path).exists():
        return []
    conn = connect(db_path)
    try:
        rows = conn.execute(
            """SELECT m.date, m.resting_hr, m.hrv_ms, m.hrv_baseline_low,
                      m.hrv_baseline_high, m.sleep_hours, m.sleep_score, m.acwr,
                      f.hrv_state, f.acwr_band, f.readiness_flag, f.sleep_debt_hours
                 FROM daily_metrics m JOIN daily_features f
                   ON m.athlete_id = f.athlete_id AND m.date = f.date
                WHERE m.athlete_id = ?
                ORDER BY m.date""",
            (athlete_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def list_athletes(db_path: str) -> list[str]:
    if not Path(db_path).exists():
        return []
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT athlete_id FROM daily_metrics ORDER BY athlete_id"
        ).fetchall()
        return [r["athlete_id"] for r in rows]
    finally:
        conn.close()


# ------------------------------------------------------------------------- sidebar
st.sidebar.title("Athlete Health RAG")
st.sidebar.caption("A context & data layer over wearable data")

db_path = str(DB_PATH)
athletes = list_athletes(db_path)

if not athletes:
    st.warning(
        "No data found. From a terminal run:\n\n"
        "```\npython -m src.cli ingest data/athlete_a_wearable_daily.csv\n"
        "python -m src.cli index\n```"
    )
    st.stop()

athlete_id = st.sidebar.selectbox("Athlete", athletes)
st.sidebar.divider()
st.sidebar.caption(
    "Offline-first demo. This is an engineering demonstration over wearable "
    "wellness data and is **not** medical advice."
)

daily = load_daily(db_path, athlete_id)
dates = [d["date"] for d in daily]


# ---------------------------------------------------------------------------- tabs
tab_trends, tab_ask, tab_eval = st.tabs(["Trends", "Ask", "Retrieval quality"])

with tab_trends:
    st.subheader(f"Training & recovery trends — {athlete_id}")
    stats = summary(db_path, athlete_id)
    ov = stats["overview"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Days", ov.get("days"))
    c2.metric("Avg resting HR", f"{ov.get('avg_rhr')} bpm")
    c3.metric("Avg HRV", f"{ov.get('avg_hrv')} ms")
    c4.metric("Avg sleep", f"{ov.get('avg_sleep')} h")

    st.markdown("**Acute:chronic workload ratio (ACWR)** — bands: <0.8 detraining, "
                "0.8-1.3 optimal, 1.3-1.5 caution, >1.5 high risk.")
    st.line_chart({d["date"]: d["acwr"] for d in daily if d["acwr"] is not None},
                  x_label="date", y_label="ACWR")

    st.markdown("**HRV vs personal baseline band** — readings below the low band "
                "suggest suppressed recovery.")
    st.line_chart(
        {
            "HRV": {d["date"]: d["hrv_ms"] for d in daily},
            "baseline_low": {d["date"]: d["hrv_baseline_low"] for d in daily},
            "baseline_high": {d["date"]: d["hrv_baseline_high"] for d in daily},
        },
        y_label="ms",
    )

    st.markdown("**Sleep hours & score**")
    st.line_chart(
        {
            "sleep_hours": {d["date"]: d["sleep_hours"] for d in daily},
            "sleep_score (÷10)": {d["date"]: (d["sleep_score"] or 0) / 10 for d in daily},
        },
    )

    st.markdown("**Flagged days**")
    caution = stats["caution_or_high_risk_days"]
    suppressed = stats["suppressed_hrv_days"]
    colA, colB = st.columns(2)
    with colA:
        st.caption("Caution / high-risk load days")
        st.dataframe(caution or [{"date": "none", "acwr": "-"}], use_container_width=True)
    with colB:
        st.caption("Suppressed-HRV days")
        st.dataframe(suppressed or [{"date": "none"}], use_container_width=True)


with tab_ask:
    st.subheader("Ask the context layer")
    st.caption("Answers are grounded in retrieved dated records and must cite them. "
               "Medical/diagnostic questions are refused by the guardrail.")

    example = st.selectbox(
        "Try an example, or type your own below",
        [
            "Was I overtraining in early September?",
            "When was my training load dangerously high?",
            "Which days was my HRV suppressed?",
            "When did I sleep the worst?",
            "Do I have an injury?  (guardrail demo)",
        ],
    )
    question = st.text_input("Question", value=example)
    top_k = st.slider("top_k", 1, 12, 6)

    if st.button("Ask", type="primary"):
        retriever = Retriever()
        if retriever.store.count() == 0:
            st.error("Vector store is empty. Run `python -m src.cli index` first.")
        else:
            pipeline = RagPipeline(retriever=retriever)
            with st.spinner("Retrieving + generating..."):
                res = pipeline.ask(question, top_k=top_k, athlete_id=athlete_id)
            badge = "guardrail triggered" if res.guard_triggered else (
                "citation ok" if res.citation_ok else "citation REJECTED")
            st.caption(f"provider: `{res.provider}`  ·  {badge}  ·  {res.citation_note}")
            st.markdown(res.answer)
            st.markdown("**Sources (retrieved dated records)**")
            st.dataframe(res.sources, use_container_width=True)


with tab_eval:
    st.subheader("Retrieval quality")
    st.caption("Precision/recall/hit-rate/MRR against a curated gold set grounded in "
               "the SQL analytics. This is how retrieval relevance is measured and "
               "iterated on.")
    k = st.slider("k", 1, 12, 6, key="eval_k")
    if st.button("Run evaluation"):
        retriever = Retriever()
        if retriever.store.count() == 0:
            st.error("Vector store is empty. Run `python -m src.cli index` first.")
        else:
            report = evaluate(retriever=retriever, k=k)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mean P@k", report.mean_precision)
            m2.metric("Mean R@k", report.mean_recall)
            m3.metric("Hit rate", report.hit_rate)
            m4.metric("MRR", report.mrr)
            st.dataframe(
                [
                    {
                        "question": s.question,
                        "P@k": s.precision_at_k,
                        "R@k": s.recall_at_k,
                        "hit": s.hit_at_k,
                        "RR": s.reciprocal_rank,
                        "retrieved": ", ".join(s.retrieved_dates[:k]),
                    }
                    for s in report.per_query
                ],
                use_container_width=True,
            )
