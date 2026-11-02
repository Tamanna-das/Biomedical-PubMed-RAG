"""BM25, dense and hybrid (reciprocal rank fusion) retrieval behind one interface."""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.config import Settings, get_settings
from src.index.bm25 import BM25Index
from src.index.dense import DenseIndex, get_encoder
from src.io_utils import read_table

log = logging.getLogger("retrieve")

METHODS = ("bm25", "dense", "hybrid")


@dataclass
class Hit:
    pmid: str
    title: str
    abstract: str
    journal: str
    year: int | None
    score: float  # score of the method that produced the ranking (RRF / BM25 / cosine / reranker)
    rank: int
    bm25_rank: int | None = None
    dense_rank: int | None = None
    dense_score: float | None = None  # cosine(query, doc); used for abstention
    rerank_score: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _year(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: score(d) = sum_r 1 / (k + rank_r(d)), ranks starting at 1."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for r, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + r)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


class Retriever:
    def __init__(self, docs: dict[str, dict], bm25: BM25Index, dense: DenseIndex | None = None,
                 encoder=None, reranker=None, rrf_k: int = 60, depth: int = 100):
        self.docs, self.bm25, self.dense, self.encoder = docs, bm25, dense, encoder
        self.reranker, self.rrf_k, self.depth = reranker, rrf_k, depth
        self._row = {pid: i for i, pid in enumerate(dense.ids)} if dense is not None else {}

    @classmethod
    def from_disk(cls, s: Settings | None = None, corpus_path: Path | None = None, index_dir: Path | None = None,
                  encoder_name: str | None = None, use_reranker: bool = False, device: str | None = None) -> "Retriever":
        s = s or get_settings()
        corpus_path = corpus_path or s.corpus_path
        index_dir = index_dir or s.index_dir
        df = read_table(corpus_path)
        df["pmid"] = df["pmid"].astype(str)
        docs = {r["pmid"]: r for r in df[["pmid", "title", "abstract", "journal", "year"]].to_dict("records")}
        bm25 = BM25Index.load(index_dir / "bm25")
        dense = encoder = None
        if (index_dir / "dense" / "meta.json").exists():
            dense = DenseIndex.load(index_dir / "dense")
            name = encoder_name or dense.encoder_name
            if name != dense.encoder_name:
                raise ValueError(f"index built with {dense.encoder_name!r} but encoder {name!r} requested")
            encoder = get_encoder(name, s, index_dir / "dense", device)
        else:
            log.warning("no dense index at %s; only BM25 is available", index_dir / "dense")
        reranker = None
        if use_reranker:
            from src.retrieve.rerank import CrossEncoderReranker

            reranker = CrossEncoderReranker(s.reranker_model, device)
        return cls(docs, bm25, dense, encoder, reranker, s.rrf_k, s.candidate_depth)

    @property
    def methods(self) -> tuple[str, ...]:
        return METHODS if self.dense is not None else ("bm25",)

    # ------------------------------------------------------------------
    def _dense_rank(self, qv: np.ndarray, depth: int) -> list[tuple[str, float]]:
        return self.dense.search_vectors(qv, depth)[0]

    def search(self, query: str, method: str = "hybrid", k: int = 5, rerank: bool = False) -> list[Hit]:
        if method not in self.methods:
            raise ValueError(f"method {method!r} unavailable; choose from {self.methods}")
        depth = max(self.depth, k)
        bm25_list = self.bm25.search(query, depth) if method in ("bm25", "hybrid") else []
        qv = self.encoder.encode_queries([query]) if self.dense is not None else None
        dense_list = self._dense_rank(qv, depth) if method in ("dense", "hybrid") else []

        if method == "bm25":
            ranked = bm25_list
        elif method == "dense":
            ranked = dense_list
        else:
            ranked = rrf([[d for d, _ in bm25_list], [d for d, _ in dense_list]], self.rrf_k)

        bm25_pos = {d: i + 1 for i, (d, _) in enumerate(bm25_list)}
        dense_pos = {d: i + 1 for i, (d, _) in enumerate(dense_list)}
        n_keep = depth if (rerank and self.reranker is not None) else k
        cands = [(d, sc) for d, sc in ranked[:n_keep] if d in self.docs]

        hits = []
        for i, (d, sc) in enumerate(cands, start=1):
            doc = self.docs[d]
            hits.append(Hit(pmid=d, title=doc["title"], abstract=doc["abstract"], journal=doc.get("journal", ""),
                            year=_year(doc.get("year")),
                            score=float(sc), rank=i, bm25_rank=bm25_pos.get(d), dense_rank=dense_pos.get(d)))
        if qv is not None and hits:  # exact cosine for every returned hit (needed for abstention)
            rows = [self._row[h.pmid] for h in hits]
            for h, c in zip(hits, (self.dense.emb[rows] @ qv[0]).tolist()):
                h.dense_score = float(c)
        if rerank and self.reranker is not None and hits:
            rs = self.reranker.score(query, [f"{h.title}. {h.abstract}" for h in hits])
            for h, r in zip(hits, rs.tolist()):
                h.rerank_score = h.score = float(r)
            hits = sorted(hits, key=lambda h: -h.rerank_score)[:k]
            for i, h in enumerate(hits, start=1):
                h.rank = i
        return hits

    def ranked_ids(self, query: str, method: str, k: int, rerank: bool = False) -> list[str]:
        return [h.pmid for h in self.search(query, method, k, rerank)]
