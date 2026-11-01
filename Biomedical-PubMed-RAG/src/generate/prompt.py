"""Prompt templates.

`grounded` is the system under test: answer only from the numbered abstracts, cite every
claim as [PMID:n], and emit the abstain token when the evidence is not there.
`plain` is the baseline it is compared against in answer_eval (same context, no rules).
"""
from __future__ import annotations

ABSTAIN_TOKEN = "INSUFFICIENT_EVIDENCE"

GROUNDED_SYSTEM = f"""You are a biomedical research assistant. You answer questions using ONLY the PubMed abstracts provided.

Rules:
1. Every factual sentence must end with at least one citation in the form [PMID:12345678], naming the abstract(s) that support it.
2. Cite only PMIDs that appear in the provided abstracts. Never use outside knowledge.
3. If the abstracts do not contain enough information to answer, reply with exactly {ABSTAIN_TOKEN} on the first line, followed by one sentence saying what is missing.
4. If abstracts disagree, say so and cite both sides.
5. Be concise: 2-6 sentences. Do not speculate beyond what the abstracts report."""

PLAIN_SYSTEM = "You are a helpful biomedical assistant. Answer the question."


def format_context(hits, max_chars: int = 12000) -> str:
    blocks, used = [], 0
    for h in hits:
        meta = ", ".join(str(x) for x in (h.journal, h.year) if x)
        block = f"[PMID:{h.pmid}] {h.title}" + (f" ({meta})" if meta else "") + f"\n{h.abstract}"
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def build_messages(question: str, hits, mode: str = "grounded", max_chars: int = 12000) -> list[dict]:
    context = format_context(hits, max_chars)
    if mode == "grounded":
        user = f"Abstracts:\n\n{context}\n\nQuestion: {question}\n\nAnswer (with [PMID:...] citations):"
        system = GROUNDED_SYSTEM
    elif mode == "plain":
        user = f"Context:\n\n{context}\n\nQuestion: {question}"
        system = PLAIN_SYSTEM
    else:
        raise ValueError(f"unknown prompt mode {mode!r}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
