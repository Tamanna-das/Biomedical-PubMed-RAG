"""Retriever + LLM in one object, shared by the API, the UI and the evaluation scripts.

    python -m src.pipeline "Which genes confer colistin resistance in Enterobacterales?"
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache

from src.config import Settings, get_settings
from src.generate.answer import Answer, generate_answer
from src.generate.llm import get_llm
from src.retrieve.hybrid import Retriever


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm, s: Settings):
        self.retriever, self.llm, self.s = retriever, llm, s

    @classmethod
    def from_settings(cls, s: Settings | None = None, use_reranker: bool = False, backend: str | None = None,
                      **retriever_kwargs) -> "RAGPipeline":
        s = s or get_settings()
        return cls(Retriever.from_disk(s, use_reranker=use_reranker, **retriever_kwargs), get_llm(s, backend), s)

    def ask(self, question: str, method: str = "hybrid", k: int | None = None, rerank: bool = False,
            mode: str = "grounded") -> Answer:
        hits = self.retriever.search(question, method, k or self.s.top_k, rerank)
        return generate_answer(question, hits, self.llm, mode, self.s.abstain_threshold, self.s.max_context_chars,
                               self.s.price_in_per_m, self.s.price_out_per_m)


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    import os

    return RAGPipeline.from_settings(use_reranker=os.environ.get("USE_RERANKER", "0") == "1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--method", default="hybrid")
    ap.add_argument("-k", type=int, default=None)
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()
    p = RAGPipeline.from_settings(backend=args.backend)
    a = p.ask(args.question, args.method, args.k)
    d = a.to_dict()
    d.pop("hits")
    print(json.dumps(d, indent=2))


if __name__ == "__main__":
    main()
