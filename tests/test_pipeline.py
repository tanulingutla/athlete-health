"""End-to-end and unit tests. Run with: python -m pytest  (or python -m unittest)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config import THRESHOLDS
from src.features import classify_acwr, classify_hrv, compute_sleep_debt, readiness
from src.guardrails import (
    enforce_citation,
    is_medical_advice_request,
    redact_pii,
)
from src.ingest import CsvSource, ingest, validate_row
from src.narrative import render_all
from src.rag import RagPipeline
from src.retriever import Retriever, infer_filter
from src.schema import DailyMetric, connect

CSV = Path(__file__).resolve().parent.parent / "data" / "athlete_a_wearable_daily.csv"


class FeatureTests(unittest.TestCase):
    def test_acwr_bands(self):
        self.assertEqual(classify_acwr(0.3, THRESHOLDS), "detraining")
        self.assertEqual(classify_acwr(1.0, THRESHOLDS), "optimal")
        self.assertEqual(classify_acwr(1.4, THRESHOLDS), "caution")
        self.assertEqual(classify_acwr(1.7, THRESHOLDS), "high_risk")
        self.assertEqual(classify_acwr(None, THRESHOLDS), "unknown")

    def test_hrv_vs_baseline(self):
        m = DailyMetric("a", "2026-09-03", 37, 73.0, 65.0, 112.0, 19, 8.4, 87,
                        0, 0, "", 639, 530, 1.2, 0.0)
        state, dist = classify_hrv(m)
        self.assertEqual(state, "normal")
        self.assertIsNotNone(dist)
        # below the low band -> suppressed
        m.hrv_ms = 50.0
        self.assertEqual(classify_hrv(m)[0], "suppressed")

    def test_sleep_debt(self):
        debt = compute_sleep_debt([6.0, 7.0, None, 8.0], THRESHOLDS)
        # (8-6)+(8-7)+(8-8) = 3.0
        self.assertEqual(debt, 3.0)

    def test_readiness(self):
        self.assertEqual(readiness("suppressed", "high_risk", 40, THRESHOLDS), "rest")
        self.assertEqual(readiness("normal", "optimal", 85, THRESHOLDS), "ok")
        self.assertEqual(readiness("normal", "caution", 85, THRESHOLDS), "monitor")


class GuardrailTests(unittest.TestCase):
    def test_redact_pii(self):
        red = redact_pii("email me at a@b.com or 555-123-4567, id 998877665")
        self.assertNotIn("a@b.com", red)
        self.assertIn("[REDACTED_EMAIL]", red)
        self.assertIn("[REDACTED_PHONE]", red)
        self.assertIn("[REDACTED_ID]", red)

    def test_medical_advice_detection(self):
        self.assertTrue(is_medical_advice_request("Do I have an injury?"))
        self.assertTrue(is_medical_advice_request("What medication should I take?"))
        self.assertFalse(is_medical_advice_request("Was my training load high in September?"))

    def test_citation_enforcement(self):
        ok, _ = enforce_citation("On 2026-09-03 load spiked.", {"2026-09-03"})
        self.assertTrue(ok)
        bad, _ = enforce_citation("Load spiked a lot.", {"2026-09-03"})
        self.assertFalse(bad)
        hallucinated, _ = enforce_citation("On 2099-01-01 load spiked.", {"2026-09-03"})
        self.assertFalse(hallucinated)


class IngestTests(unittest.TestCase):
    def test_validation_rejects_bad_rows(self):
        m, err = validate_row({"date": "not-a-date"}, "a")
        self.assertIsNone(m)
        self.assertIn("bad date", err)
        m, err = validate_row({"date": "2026-01-01", "resting_hr": "999"}, "a")
        self.assertIsNone(m)
        self.assertIn("out of range", err)

    def test_dedupe_on_reingest(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "t.db"
            r1 = ingest(CsvSource(CSV), "athlete-a", db)
            self.assertGreater(r1.rows_written, 0)
            r2 = ingest(CsvSource(CSV), "athlete-a", db)  # identical re-ingest
            self.assertEqual(r2.rows_written, 0)
            self.assertEqual(r2.rows_skipped_dupe, r1.rows_written)


class RetrievalIntentTests(unittest.TestCase):
    def test_intent_filter(self):
        self.assertEqual(infer_filter("was I overtraining?"),
                         {"acwr_band": {"$in": ["caution", "high_risk"]}})
        self.assertIsNone(infer_filter("tell me about tuesdays"))


class EmbeddingDeterminismTests(unittest.TestCase):
    def test_hashing_embedder_is_process_stable(self):
        """Embeddings must not depend on PYTHONHASHSEED, or index-time and query-time
        vectors would disagree across processes and silently break retrieval."""
        from src.embeddings import HashingEmbedder

        v1 = HashingEmbedder().embed(["suppressed hrv on a rest day"])[0]
        v2 = HashingEmbedder().embed(["suppressed hrv on a rest day"])[0]
        self.assertEqual(v1, v2)
        # A known token must always land in the same bucket.
        from src.embeddings import _stable_hash

        self.assertEqual(_stable_hash("overtraining"), _stable_hash("overtraining"))


class EvaluationTests(unittest.TestCase):
    def test_retrieval_quality_floor(self):
        """The gold set should at least be fully hit (every query finds a relevant
        record) even with the weak offline hashing embedder."""
        from src.evaluate import evaluate
        from src.narrative import render_all
        from src.vectorstore import JsonVectorStore

        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "t.db"
            store = Path(d) / "vec.json"
            ingest(CsvSource(CSV), "athlete-a", db)
            conn = connect(db)
            docs = render_all(conn, "athlete-a")
            conn.close()
            retriever = Retriever(store=JsonVectorStore(store))
            retriever.index(docs)

            report = evaluate(retriever=retriever, k=6)
            # Every gold query must surface at least one relevant date.
            self.assertEqual(report.hit_rate, 1.0)
            # And the first relevant hit should rank highly on average.
            self.assertGreaterEqual(report.mrr, 0.8)


class EndToEndTests(unittest.TestCase):
    def test_full_pipeline_offline(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "t.db"
            store = Path(d) / "vec.json"
            ingest(CsvSource(CSV), "athlete-a", db)

            conn = connect(db)
            docs = render_all(conn, "athlete-a")
            conn.close()
            self.assertEqual(len(docs), 56)
            self.assertIn("ACWR", docs[0]["text"])

            from src.vectorstore import JsonVectorStore

            retriever = Retriever(store=JsonVectorStore(store))
            self.assertEqual(retriever.index(docs), 56)

            pipeline = RagPipeline(retriever=retriever, provider="offline")
            res = pipeline.ask("Was I overtraining in early September?")
            self.assertTrue(res.answer)
            self.assertTrue(res.sources)
            self.assertEqual(res.provider, "offline")

    def test_medical_guard_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "t.db"
            store = Path(d) / "vec.json"
            ingest(CsvSource(CSV), "athlete-a", db)
            conn = connect(db)
            docs = render_all(conn, "athlete-a")
            conn.close()
            from src.vectorstore import JsonVectorStore

            retriever = Retriever(store=JsonVectorStore(store))
            retriever.index(docs)
            pipeline = RagPipeline(retriever=retriever, provider="offline")
            res = pipeline.ask("Do I have an injury?")
            self.assertTrue(res.guard_triggered)
            self.assertIn("can't provide medical", res.answer.lower())


if __name__ == "__main__":
    unittest.main()
