"""Command-line entry points: ingest, index, ask, stats.

Run as a module:  python -m src.cli <command> ...
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from datetime import date

from .analytics import summary
from .cohort import generate_default_cohort
from .config import DB_PATH, PROVIDERS
from .evaluate import evaluate
from .ingest import CsvSource, ingest
from .narrative import render_all
from .rag import RagPipeline
from .retriever import Retriever
from .schema import connect


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


# --------------------------------------------------------------------- commands
def cmd_ingest(args) -> int:
    source = CsvSource(args.csv)
    report = ingest(source, athlete_id=args.athlete_id, db_path=args.db)
    print(f"Ingested from {source.name}")
    print(f"  seen={report.rows_seen} written={report.rows_written} "
          f"dupes={report.rows_skipped_dupe} rejected={report.rows_rejected}")
    print(f"  date range: {report.earliest_date} -> {report.latest_date}")
    if report.rejected_detail:
        print("  rejections:")
        for r in report.rejected_detail:
            print(f"    - {r}")
    if report.gaps:
        print(f"  freshness: {len(report.gaps)} missing day(s): {', '.join(report.gaps)}")
    else:
        print("  freshness: no gaps in the ingested range")
    return 0


def cmd_index(args) -> int:
    conn = connect(args.db)
    try:
        docs = render_all(conn, args.athlete_id)
    finally:
        conn.close()
    if not docs:
        print("No data to index. Run `ingest` first.", file=sys.stderr)
        return 1
    retriever = Retriever(embed_backend=args.embed_backend,
                          force_fallback=args.force_fallback)
    n = retriever.index(docs)
    print(f"Indexed {n} daily narratives using "
          f"embedder='{retriever.embedder.backend}', store='{retriever.store.backend}'.")
    # Show one sample narrative so reviewers can see what gets embedded.
    print("\nSample narrative:\n  " + docs[len(docs) // 2]["text"])
    return 0


def cmd_ask(args) -> int:
    retriever = Retriever(embed_backend=args.embed_backend,
                          force_fallback=args.force_fallback)
    if retriever.store.count() == 0:
        print("Vector store is empty. Run `index` first.", file=sys.stderr)
        return 1
    pipeline = RagPipeline(retriever=retriever, provider=args.provider)
    result = pipeline.ask(args.question, top_k=args.top_k, athlete_id=args.athlete_id)

    print(f"Q: {result.question}")
    print(f"[provider={result.provider}] "
          f"[citation: {'ok' if result.citation_ok else 'REJECTED'} - {result.citation_note}]")
    print()
    print(result.answer)
    print("\nSources (retrieved dated records):")
    for s in result.sources:
        print(f"  - {s['date']}  acwr={s['acwr_band']}  readiness={s['readiness_flag']}  "
              f"score={s['score']}")
    return 0


def cmd_stats(args) -> int:
    data = summary(args.db, args.athlete_id)
    if args.json:
        print(json.dumps(data, indent=2))
        return 0

    ov = data["overview"]
    print(f"Overview for {args.athlete_id}:")
    print(f"  {ov.get('days')} days  ({ov.get('first_day')} -> {ov.get('last_day')})")
    print(f"  avg resting HR {ov.get('avg_rhr')} bpm | avg HRV {ov.get('avg_hrv')} ms")
    print(f"  avg sleep {ov.get('avg_sleep')} h (score {ov.get('avg_sleep_score')}) | "
          f"total exercise {ov.get('total_exercise_min')} min")

    print("\nACWR injury-risk bands:")
    for b in data["acwr_band_counts"]:
        print(f"  {b['acwr_band']:<11} {b['n']} day(s)")

    print("\nCaution / high-risk load days:")
    for d in data["caution_or_high_risk_days"]:
        print(f"  {d['date']}  acwr={d['acwr']}  readiness={d['readiness_flag']}")

    print("\nSuppressed-HRV days (under-recovery):")
    for d in data["suppressed_hrv_days"]:
        print(f"  {d['date']}  {d['hrv_vs_baseline']:+.1f} ms vs baseline midpoint")

    print("\nWorst sleep nights:")
    for d in data["worst_sleep_days"]:
        print(f"  {d['date']}  {d['sleep_hours']} h  score {d['sleep_score']}")

    print("\nReadiness distribution:")
    for r in data["readiness_counts"]:
        print(f"  {r['readiness_flag']:<8} {r['n']} day(s)")
    return 0


def cmd_evaluate(args) -> int:
    retriever = Retriever(embed_backend=args.embed_backend,
                          force_fallback=args.force_fallback)
    if retriever.store.count() == 0:
        print("Vector store is empty. Run `index` first.", file=sys.stderr)
        return 1
    report = evaluate(retriever=retriever, k=args.top_k, athlete_id=args.athlete_id)
    print(f"Retrieval evaluation (athlete={args.athlete_id}, k={report.k}, "
          f"embedder='{retriever.embedder.backend}')")
    print(f"{'question':<52}{'P@k':>7}{'R@k':>7}{'hit':>6}{'RR':>7}")
    print("-" * 79)
    for s in report.per_query:
        q = (s.question[:49] + "...") if len(s.question) > 52 else s.question
        print(f"{q:<52}{s.precision_at_k:>7.2f}{s.recall_at_k:>7.2f}"
              f"{s.hit_at_k:>6.0f}{s.reciprocal_rank:>7.2f}")
    print("-" * 79)
    print(f"{'MACRO AVERAGE':<52}{report.mean_precision:>7.2f}"
          f"{report.mean_recall:>7.2f}{report.hit_rate:>6.2f}{report.mrr:>7.2f}")
    return 0


def cmd_cohort(args) -> int:
    created = generate_default_cohort(days=args.days)
    print("Generated synthetic cohort CSVs:")
    for athlete_id, path in created:
        print(f"  {athlete_id}: {path}")
    print("\nNext: ingest and index each, e.g.")
    for athlete_id, path in created:
        print(f"  python -m src.cli ingest {path} --athlete-id {athlete_id} && "
              f"python -m src.cli index --athlete-id {athlete_id}")
    return 0


# ------------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="athlete-rag",
                                description="Athlete Health RAG context & data layer.")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--db", default=str(DB_PATH), help="SQLite path")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("ingest", help="Ingest a wearable CSV into the storage layer.")
    pi.add_argument("csv")
    pi.add_argument("--athlete-id", default="athlete-a")
    pi.set_defaults(func=cmd_ingest)

    px = sub.add_parser("index", help="Build the semantic index from stored data.")
    px.add_argument("--athlete-id", default="athlete-a")
    px.add_argument("--embed-backend", default=None,
                    choices=["sentence_transformers", "openai", "hashing"])
    px.add_argument("--force-fallback", action="store_true",
                    help="Force the pure-python vector store.")
    px.set_defaults(func=cmd_index)

    pa = sub.add_parser("ask", help="Ask a question against the indexed context.")
    pa.add_argument("question")
    pa.add_argument("--athlete-id", default="athlete-a")
    pa.add_argument("--top-k", type=int, default=None)
    pa.add_argument("--provider", default=None,
                    choices=["openai", "azure", "anthropic", "offline"])
    pa.add_argument("--embed-backend", default=None,
                    choices=["sentence_transformers", "openai", "hashing"])
    pa.add_argument("--force-fallback", action="store_true")
    pa.set_defaults(func=cmd_ask)

    ps = sub.add_parser("stats", help="Run SQL analytics over the stored data.")
    ps.add_argument("--athlete-id", default="athlete-a")
    ps.add_argument("--json", action="store_true")
    ps.set_defaults(func=cmd_stats)

    pe = sub.add_parser("evaluate", help="Measure retrieval quality against a gold set.")
    pe.add_argument("--athlete-id", default="athlete-a")
    pe.add_argument("--top-k", type=int, default=6)
    pe.add_argument("--embed-backend", default=None,
                    choices=["sentence_transformers", "openai", "hashing"])
    pe.add_argument("--force-fallback", action="store_true")
    pe.set_defaults(func=cmd_evaluate)

    pc = sub.add_parser("cohort", help="Generate synthetic multi-athlete CSVs.")
    pc.add_argument("--days", type=int, default=56)
    pc.set_defaults(func=cmd_cohort)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
