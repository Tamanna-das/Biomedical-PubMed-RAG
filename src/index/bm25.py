"""Okapi BM25 over a sparse term matrix.

Same scoring as rank-bm25's BM25Okapi (k1, b, idf = ln(1 + (N - df + 0.5) / (df + 0.5))),
but the per-document term weights are precomputed into a scipy CSC matrix, so a query is
one sparse mat-vec instead of a Python loop over 50k documents.

    python -m src.index.bm25              # build data/index/bm25/
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from src.config import get_settings
from src.io_utils import read_table, setup_logging

log = logging.getLogger("bm25")

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset("""a an and are as at be been but by can could did do does for from had has have
how however if in into is it its may might more most no not of on or our such than that the their
then there these they this those to was we were what when where which while who whom why will with
within would also between both during each other over per some study studies using used""".split())


def _light_stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    return [_light_stem(t) for t in TOKEN_RE.findall(text.lower())
            if t not in STOPWORDS and not (len(t) == 1 and t.isalpha())]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vocab: dict[str, int] = {}
        self.ids: list[str] = []
        self.weights: sp.csc_matrix | None = None  # docs x terms

    def fit(self, texts: list[str], ids: list[str]) -> "BM25Index":
        cv = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, dtype=np.float32)
        tf = cv.fit_transform(texts).tocsr()  # docs x terms
        self.vocab = {t: int(i) for t, i in cv.vocabulary_.items()}
        self.ids = list(ids)
        n_docs = tf.shape[0]
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = dl.mean() if n_docs else 1.0
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        # per-nonzero denominator: tf + k1 * (1 - b + b * dl/avgdl)
        norm = (self.k1 * (1 - self.b + self.b * dl / avgdl)).astype(np.float32)
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        data = tf.data * (self.k1 + 1) / (tf.data + norm[rows]) * idf[tf.indices]
        self.weights = sp.csr_matrix((data, tf.indices, tf.indptr), shape=tf.shape).tocsc()
        log.info("BM25 fitted: %d docs, %d terms", n_docs, len(self.vocab))
        return self

    def scores(self, query: str) -> np.ndarray:
        q = np.zeros(len(self.vocab), dtype=np.float32)
        for tok in tokenize(query):
            j = self.vocab.get(tok)
            if j is not None:
                q[j] += 1.0
        return self.weights @ q

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        s = self.scores(query)
        k = min(k, len(s))
        if k == 0:
            return []
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        return [(self.ids[i], float(s[i])) for i in top if s[i] > 0]

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        sp.save_npz(d / "weights.npz", self.weights)
        (d / "meta.json").write_text(json.dumps({"k1": self.k1, "b": self.b, "ids": self.ids}))
        (d / "vocab.json").write_text(json.dumps(self.vocab))

    @classmethod
    def load(cls, directory: str | Path) -> "BM25Index":
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        idx = cls(meta["k1"], meta["b"])
        idx.ids = meta["ids"]
        idx.vocab = json.loads((d / "vocab.json").read_text())
        idx.weights = sp.load_npz(d / "weights.npz").tocsc()
        return idx


def build(corpus_path: Path, out_dir: Path) -> BM25Index:
    df = read_table(corpus_path)
    idx = BM25Index().fit(df["text"].tolist(), df["pmid"].astype(str).tolist())
    idx.save(out_dir)
    log.info("saved -> %s", out_dir)
    return idx


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=s.corpus_path)
    ap.add_argument("--out", type=Path, default=s.index_dir / "bm25")
    args = ap.parse_args(argv)
    build(args.corpus, args.out)


if __name__ == "__main__":
    main()
