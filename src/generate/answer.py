"""Answer generation with abstention and citation checks.

Two abstention paths:
  * retrieval gate  - best hit's cosine below ABSTAIN_THRESHOLD -> abstain without calling the LLM
  * model abstains  - the LLM emits INSUFFICIENT_EVIDENCE

Post-checks (automatic, every answer):
  * invalid_citations  - cited PMIDs that were not in the retrieved context (hallucinated)
  * uncited_sentences  - sentences with no citation (a proxy for unsupported claims)
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

from src.generate.prompt import ABSTAIN_TOKEN, build_messages

CITE_RE = re.compile(r"PMID\s*:?\s*(\d{5,9})", re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


@dataclass
class Answer:
    question: str
    answer: str
    abstained: bool
    abstain_reason: str | None
    citations: list[str]
    invalid_citations: list[str]
    context_pmids: list[str]
    n_sentences: int
    uncited_sentences: int
    model: str
    mode: str
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    hits: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def extract_citations(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for m in CITE_RE.findall(text):
        seen.setdefault(m, None)
    return list(seen)


def sentence_stats(text: str) -> tuple[int, int]:
    """(number of sentences, sentences without a PMID citation). Headers/short fragments ignored."""
    sents = [s for s in SENT_SPLIT.split(text.strip()) if len(s.split()) >= 4]
    return len(sents), sum(1 for s in sents if not CITE_RE.search(s))


def generate_answer(question: str, hits, llm, mode: str = "grounded", abstain_threshold: float = 0.0,
                    max_context_chars: int = 12000, price_in: float = 0.0, price_out: float = 0.0) -> Answer:
    t0 = time.time()
    ctx = [h.pmid for h in hits]
    hit_dicts = [h.to_dict() for h in hits]
    best = max((h.dense_score for h in hits if h.dense_score is not None), default=None)
    gate = not hits or (best is not None and best < abstain_threshold)
    if gate and mode == "grounded":
        reason = "no documents retrieved" if not hits else f"best retrieval cosine {best:.3f} < {abstain_threshold}"
        return Answer(question, f"{ABSTAIN_TOKEN}\nNot enough evidence in the corpus ({reason}).", True,
                      "retrieval_gate", [], [], ctx, 0, 0, getattr(llm, "model", "?"), mode,
                      round(time.time() - t0, 3), hits=hit_dicts)

    resp = llm.generate(build_messages(question, hits, mode, max_context_chars))
    text = resp.text.strip()
    abstained = text.upper().startswith(ABSTAIN_TOKEN) or ABSTAIN_TOKEN in text.upper()[:200]
    cites = extract_citations(text)
    ctx_set = set(ctx)
    n_sent, uncited = (0, 0) if abstained else sentence_stats(text)
    cost = resp.input_tokens / 1e6 * price_in + resp.output_tokens / 1e6 * price_out
    return Answer(question, text, abstained, "model" if abstained else None, cites,
                  [c for c in cites if c not in ctx_set], ctx, n_sent, uncited, resp.model, mode,
                  round(time.time() - t0, 3), resp.input_tokens, resp.output_tokens, round(cost, 6), hit_dicts)
