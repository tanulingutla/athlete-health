"""Vendor-agnostic chat client with an offline fallback.

A single `generate(system, user)` call routes to OpenAI, Azure OpenAI, or Anthropic
based on config resolution, and falls back to a deterministic extractive answerer when
no provider is configured — so the RAG demo produces a sensible, cited answer with no
API key at all.
"""
from __future__ import annotations

import logging
import re

from .config import PROVIDERS

log = logging.getLogger("llm")


class LLMClient:
    def __init__(self, provider: str | None = None):
        self.provider = PROVIDERS.resolve_llm(provider)

    def generate(self, system: str, user: str) -> str:
        try:
            if self.provider == "openai":
                return self._openai(system, user)
            if self.provider == "azure":
                return self._azure(system, user)
            if self.provider == "anthropic":
                return self._anthropic(system, user)
        except Exception as exc:  # pragma: no cover - network path
            log.warning("provider %s failed (%s); using offline answerer", self.provider, exc)
            self.provider = "offline"
        return self._offline(system, user)

    # ---------------------------------------------------------------- providers
    def _openai(self, system: str, user: str) -> str:
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=PROVIDERS.openai_model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()

    def _azure(self, system: str, user: str) -> str:
        import os

        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
        )
        resp = client.chat.completions.create(
            model=PROVIDERS.azure_deployment,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()

    def _anthropic(self, system: str, user: str) -> str:
        import anthropic

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=PROVIDERS.anthropic_model,
            max_tokens=700,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content).strip()

    # ----------------------------------------------------------------- offline
    def _offline(self, system: str, user: str) -> str:
        """Deterministic extractive answerer.

        It does not invent content: it stitches together the retrieved context
        snippets (which already carry dated, interpreted signal) into a short,
        grounded summary. This keeps the demo honest and fully offline.
        """
        context_block = ""
        m = re.search(r"CONTEXT:\s*(.*?)\s*QUESTION:", user, re.DOTALL)
        if m:
            context_block = m.group(1).strip()

        lines = [ln.strip() for ln in context_block.splitlines() if ln.strip()]
        snippets = [ln for ln in lines if ln.startswith("[")]
        if not snippets:
            return ("I don't have enough indexed context to answer that. "
                    "Try ingesting and indexing more data first. (not medical advice)")

        head = snippets[:4]
        summary = " ".join(head)
        return (
            "Based on the retrieved daily records:\n\n"
            f"{summary}\n\n"
            "These dated observations are the grounding for this summary. "
            "This is an engineering demo and not medical advice."
        )
