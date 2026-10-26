# Athlete Health RAG — a context & data layer over wearable data

A retrieval-augmented generation (RAG) system that turns raw daily wearable-device
exports (resting HR, HRV, sleep, training load) into a **queryable context layer**, then
answers natural-language questions about an athlete's readiness, training load, and
recovery — with **grounded, cited answers** and **governance guardrails**.

Built to demonstrate the full context/data-layer stack an AI engineer owns in a
regulated, data-intensive health setting:

- **Ingestion** — pull daily records from a source (CSV today, a live Garmin/Apple Health
  API adapter stub with retry/backoff for tomorrow), validate, de-duplicate, and track
  freshness.
- **Storage / schema layer** — a typed SQLite schema for structured metrics plus derived
  sports-science features (HRV-vs-baseline, sleep debt, ACWR injury-risk band).
- **Semantic index** — each day is rendered to a natural-language "daily narrative",
  chunked, embedded, and written to a vector store (ChromaDB when available, a pure-Python
  fallback otherwise).
- **Retrieval** — hybrid semantic + structured filtering, tunable `top_k`.
- **Generation** — a vendor-agnostic LLM client (OpenAI / Azure OpenAI / Anthropic) with
  a deterministic offline fallback so the demo runs with **no API key**.
- **Governance** — citation enforcement, PII redaction, a "not medical advice" guard, and
  an append-only audit log of every query.

## Why this design

Wearable streams are messy: gaps, duplicate pulls, and drifting baselines. The value is
not the raw numbers — it is the *context*. ACWR (acute:chronic workload ratio) flags
injury risk; HRV read against a **personal rolling baseline** (not a population average)
flags under-recovery; sleep debt compounds. This project encodes that domain knowledge
into the retrieval layer so the LLM answers from grounded, pre-computed signal instead of
guessing.

## Quick start

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt            # core deps only; runs fully offline

# 1. Ingest the CSV into the SQLite store (validate + dedupe + derive features)
python -m src.cli ingest data/athlete_a_wearable_daily.csv --athlete-id athlete-a

# 2. Build the semantic index (daily narratives -> embeddings -> vector store)
python -m src.cli index --athlete-id athlete-a

# 3. Ask questions
python -m src.cli ask "Was I overtraining in early September?"
python -m src.cli ask "How was my sleep the week of the race taper?"

# 4. Run SQL analytics directly (no LLM)
python -m src.cli stats --athlete-id athlete-a

# 5. Measure retrieval quality against a curated gold set
python -m src.cli evaluate

# 6. Generate a synthetic multi-athlete cohort (distinct training profiles)
python -m src.cli cohort
python -m src.cli ingest data/athlete-b_wearable_daily.csv --athlete-id athlete-b
python -m src.cli index --athlete-id athlete-b
```

### Interactive dashboard

```bash
pip install streamlit
streamlit run app.py
```

Three panes: **Trends** (ACWR / HRV-vs-baseline / sleep charts straight from SQLite),
**Ask** (chat over the RAG pipeline with cited sources and the guardrail demo), and
**Retrieval quality** (the evaluation harness scores, live).

### Retrieval evaluation

`python -m src.cli evaluate` scores retrieval against a gold set of question → expected-date
pairs grounded in the SQL analytics, reporting precision@k, recall@k, hit-rate, and MRR.
This is the "iterate on context relevance and accuracy" loop made measurable. With the
offline hashing embedder the harness reports a strong-but-honest baseline (hit-rate 1.0,
MRR ~0.88, precision ~0.88): three of four gold queries retrieve perfectly, and the
"worst sleep" query retrieves poor-sleep days but can't perfectly rank the five *worst*
within them — a documented ceiling of the zero-dependency embedder. Installing
`sentence-transformers` raises precision without any code change, since the retrieval
plumbing is identical across backends.

A results notebook, `analysis.ipynb`, charts the real findings (the Sept 3 load spike,
the late-Sept HRV suppression cluster, the Oct 11/19 sleep crash) with matplotlib and
ends with the live retrieval-quality scores. Needs `pip install matplotlib`.

### Enable real LLMs / ChromaDB (optional)

```bash
pip install -r requirements-llm.txt        # openai / anthropic / chromadb / sentence-transformers
setx OPENAI_API_KEY "sk-..."               # or AZURE_OPENAI_* / ANTHROPIC_API_KEY
python -m src.cli ask "..." --provider openai
```

The system auto-detects what is installed and configured and degrades gracefully. See
`config.py` for the resolution order.

## Architecture

```
CSV / live API --> ingest --> [SQLite: daily_metrics + derived features]
                                        |
                                 render daily narrative
                                        |
                                 chunk + embed
                                        |
                              [Vector store: Chroma | fallback]
                                        |
   question --> retrieve (semantic + structured filter) --> prompt + guardrails
                                        |
                          LLM (OpenAI/Azure/Anthropic | offline) --> cited answer
                                        |
                                   audit log
```

## Project layout

```
src/
  config.py          # provider/model resolution, paths, thresholds
  schema.py          # SQLite DDL + typed row model
  features.py        # sports-science derivations (HRV baseline, sleep debt, ACWR band)
  ingest.py          # CSV + live-API adapter, validation, dedupe, freshness
  narrative.py       # day -> natural-language chunk
  embeddings.py      # embedding backends (ST / OpenAI / hashing fallback)
  vectorstore.py     # Chroma | pure-python cosine store
  retriever.py       # hybrid retrieval
  llm.py             # vendor-agnostic chat client + offline fallback
  guardrails.py      # PII redaction, medical-advice guard, citation check
  rag.py             # orchestration
  analytics.py       # SQL analytics
  evaluate.py        # retrieval evaluation harness (precision/recall/MRR)
  cohort.py          # synthetic multi-athlete generator
  cli.py             # command-line entry points
app.py               # Streamlit dashboard
tests/
data/
```

## Data governance notes

This is wearable *wellness* data, not clinical EHR data, but the project treats it with
the same guardrails a regulated environment would require:

- Every generated answer must cite the dates it drew from; uncited claims are rejected.
- Free-text fields are PII-redacted before they reach any external LLM.
- Answers carry a non-diagnostic disclaimer; the system never issues medical advice.
- Every query, its retrieved context, and the model/provider used are written to an
  append-only audit log (`audit.log`).

## Not medical advice

This project is an engineering demonstration. Outputs describe patterns in training/recovery
data and are **not** medical, diagnostic, or training advice.
