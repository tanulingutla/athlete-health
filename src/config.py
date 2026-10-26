"""Central configuration: paths, provider/model resolution, and tunable thresholds.

Everything the rest of the system needs to decide *which* backend to use lives here,
so the resolution order is in one place and easy to review.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = ROOT / "athlete_health.db"
VECTOR_STORE_PATH = ROOT / "vector_store.json"   # used by the fallback store
CHROMA_DIR = ROOT / ".chroma"
AUDIT_LOG_PATH = ROOT / "audit.log"


# ----------------------------------------------------------------- sports-science
@dataclass(frozen=True)
class SportsScienceThresholds:
    """Evidence-informed bands used to turn raw numbers into interpretable signal.

    These are intentionally conservative, documented constants rather than magic
    numbers scattered through the code. They are wellness heuristics, not clinical
    cutoffs.
    """
    # Acute:Chronic Workload Ratio. The 0.8-1.3 "sweet spot" and >1.5 danger zone
    # are widely cited in load-management literature (Gabbett and others).
    acwr_undertrained_below: float = 0.8
    acwr_optimal_high: float = 1.3
    acwr_high_risk_above: float = 1.5

    # Sleep. Debt accrues against a personal target.
    sleep_target_hours: float = 8.0
    sleep_poor_score_below: int = 60

    # HRV is judged against the athlete's own rolling baseline band, not a constant.
    # A reading below the baseline-low band suggests suppressed recovery.
    hrv_baseline_window_days: int = 7


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 6
    chunk_strategy: str = "per_day"   # per_day | per_week
    min_similarity: float = 0.0       # keep all above this cosine score


@dataclass
class ProviderConfig:
    """Resolves which embedding + LLM backend to use from the environment.

    Resolution is lazy and defensive: we only *claim* a provider is available if its
    package imports and its credentials are present. Callers can force a provider.
    """
    embedding_backend: str = field(default="")
    llm_provider: str = field(default="")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    azure_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    def resolve_llm(self, forced: str | None = None) -> str:
        """Return the LLM provider id to use: openai | azure | anthropic | offline."""
        if forced:
            return forced
        if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
            return "azure"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        return "offline"

    def resolve_embedding(self, forced: str | None = None) -> str:
        """Return the embedding backend id: sentence_transformers | openai | hashing."""
        if forced:
            return forced
        if _module_available("sentence_transformers"):
            return "sentence_transformers"
        if os.getenv("OPENAI_API_KEY") and _module_available("openai"):
            return "openai"
        return "hashing"


def _module_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


THRESHOLDS = SportsScienceThresholds()
RETRIEVAL = RetrievalConfig()
PROVIDERS = ProviderConfig()
