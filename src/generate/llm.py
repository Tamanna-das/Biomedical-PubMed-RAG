"""LLM backends.

  ollama      local models via Ollama (free; `ollama pull llama3.1:8b`)
  openai      any OpenAI-compatible chat API: Groq (free tier, the default base URL),
              OpenAI, Together, OpenRouter, a local vLLM server ...
  extractive  no LLM: picks the abstract sentences that best overlap the question and
              cites them. Deterministic; used for tests, the offline demo, and as a floor.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import requests

from src.config import Settings
from src.generate.prompt import ABSTAIN_TOKEN


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class OllamaLLM:
    def __init__(self, url: str, model: str, temperature: float = 0.0, timeout: int = 300):
        self.url, self.model, self.temperature, self.timeout = url.rstrip("/"), model, temperature, timeout

    def generate(self, messages: list[dict]) -> LLMResponse:
        r = requests.post(f"{self.url}/api/chat", timeout=self.timeout, json={
            "model": self.model, "messages": messages, "stream": False,
            "options": {"temperature": self.temperature}})
        r.raise_for_status()
        d = r.json()
        return LLMResponse(d["message"]["content"], self.model, d.get("prompt_eval_count", 0), d.get("eval_count", 0))


class OpenAICompatLLM:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.0, timeout: int = 120):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is empty; set it in .env or use LLM_BACKEND=ollama")
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model
        self.temperature, self.timeout = temperature, timeout

    def generate(self, messages: list[dict], max_retries: int = 6) -> LLMResponse:
        for attempt in range(max_retries):
            r = requests.post(f"{self.base_url}/chat/completions", timeout=self.timeout,
                              headers={"Authorization": f"Bearer {self.api_key}"},
                              json={"model": self.model, "messages": messages, "temperature": self.temperature})
            if r.status_code not in (429, 500, 502, 503) or attempt == max_retries - 1:
                break
            # free tiers rate-limit per minute: honour Retry-After, else back off
            try:
                wait = float(r.headers.get("retry-after", 0))
            except ValueError:
                wait = 0.0
            time.sleep(min(max(wait, 2 ** attempt), 60))
        r.raise_for_status()
        d = r.json()
        u = d.get("usage") or {}
        return LLMResponse(d["choices"][0]["message"]["content"], self.model,
                           u.get("prompt_tokens", 0), u.get("completion_tokens", 0))


_BLOCK_RE = re.compile(r"^\[PMID:(\d+)\][^\n]*\n(.*?)(?=\n\n\[PMID:|\n\nQuestion:|\Z)", re.S | re.M)
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


class ExtractiveLLM:
    """Selects up to `max_sents` abstract sentences covering the most question terms."""

    def __init__(self, max_sents: int = 3, min_coverage: float = 0.5):
        self.max_sents, self.min_coverage = max_sents, min_coverage
        self.model = "extractive"

    def generate(self, messages: list[dict]) -> LLMResponse:
        from src.index.bm25 import tokenize

        user = messages[-1]["content"]
        q = user.rsplit("Question:", 1)[-1].split("\n")[0]
        q_terms = set(tokenize(q))
        cands = []
        for pmid, abstract in _BLOCK_RE.findall(user):
            for sent in _SENT_RE.split(abstract.strip()):
                sent = re.sub(r"^[A-Z ]{3,}:\s*", "", sent).strip()  # drop "RESULTS:" labels
                terms = set(tokenize(sent))
                if not q_terms or len(sent) < 30:
                    continue
                cov = len(q_terms & terms) / len(q_terms)
                cands.append((cov, pmid, sent))
        cands.sort(key=lambda c: -c[0])
        chosen, seen = [], set()
        for cov, pmid, sent in cands:
            if cov < self.min_coverage or len(chosen) >= self.max_sents:
                break
            if sent not in seen:
                seen.add(sent)
                chosen.append(f"{sent.rstrip('.')} [PMID:{pmid}].")
        text = " ".join(chosen) if chosen else f"{ABSTAIN_TOKEN}\nNo abstract sentence covers the question terms."
        return LLMResponse(text, self.model, _approx_tokens(user), _approx_tokens(text))


def get_llm(s: Settings, backend: str | None = None):
    backend = backend or s.llm_backend
    if backend == "ollama":
        return OllamaLLM(s.ollama_url, s.ollama_model)
    if backend == "openai":
        return OpenAICompatLLM(s.openai_base_url, s.openai_api_key, s.openai_model)
    if backend == "extractive":
        return ExtractiveLLM()
    raise ValueError(f"unknown LLM backend {backend!r}")
