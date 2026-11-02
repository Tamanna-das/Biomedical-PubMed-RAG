"""Dense retrieval: biomedical encoders + an exact inner-product index.

Encoders
  medcpt  NCBI MedCPT (separate query / article encoders, trained on PubMed search logs).
          Articles are encoded as [title, abstract] pairs, queries with max_length 64,
          CLS pooling -- as on the model card. Recommended.
  sbert   any sentence-transformers model (default S-PubMedBert-MS-MARCO, a BioBERT-family
          model fine-tuned for retrieval).
  lsa     TF-IDF + truncated SVD. CPU-only, no downloads; used in tests and the offline demo.

All vectors are L2-normalised, so scores are cosine similarities (this makes the
abstention threshold interpretable). Index = FAISS IndexFlatIP when faiss is installed,
otherwise the same exact search in numpy -- at <=50k docs exact search takes milliseconds.

    python -m src.index.dense --encoder medcpt --device cuda   # build data/index/dense/
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from src.config import Settings, get_settings
from src.io_utils import read_table, setup_logging

log = logging.getLogger("dense")


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class Encoder(Protocol):
    name: str

    def encode_queries(self, queries: list[str]) -> np.ndarray: ...

    def encode_docs(self, titles: list[str], abstracts: list[str]) -> np.ndarray: ...


# ---------------------------------------------------------------- encoders

def _device(device: str | None) -> str:
    if device:
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MedCPTEncoder:
    name = "medcpt"

    def __init__(self, query_model: str, article_model: str, device: str | None = None, batch_size: int = 32):
        from transformers import AutoModel, AutoTokenizer

        self.device = _device(device)
        self.batch_size = batch_size
        self.q_tok = AutoTokenizer.from_pretrained(query_model)
        self.q_model = AutoModel.from_pretrained(query_model).to(self.device).eval()
        self._a = (article_model,)
        self.a_tok = self.a_model = None  # article encoder loaded lazily (only needed to build the index)

    def _load_article(self):
        if self.a_model is None:
            from transformers import AutoModel, AutoTokenizer

            self.a_tok = AutoTokenizer.from_pretrained(self._a[0])
            self.a_model = AutoModel.from_pretrained(self._a[0]).to(self.device).eval()

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(queries), self.batch_size):
                enc = self.q_tok(queries[i:i + self.batch_size], truncation=True, padding=True,
                                 return_tensors="pt", max_length=64).to(self.device)
                out.append(self.q_model(**enc).last_hidden_state[:, 0, :].float().cpu().numpy())
        return _l2(np.vstack(out))

    def encode_docs(self, titles: list[str], abstracts: list[str]) -> np.ndarray:
        import torch

        self._load_article()
        out = []
        with torch.no_grad():
            for i in range(0, len(titles), self.batch_size):
                pairs = [[t, a] for t, a in zip(titles[i:i + self.batch_size], abstracts[i:i + self.batch_size])]
                enc = self.a_tok(pairs, truncation=True, padding=True, return_tensors="pt",
                                 max_length=512).to(self.device)
                out.append(self.a_model(**enc).last_hidden_state[:, 0, :].float().cpu().numpy())
        return _l2(np.vstack(out))


class SBERTEncoder:
    name = "sbert"

    def __init__(self, model: str, device: str | None = None, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, device=_device(device))
        self.batch_size = batch_size

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return _l2(self.model.encode(queries, batch_size=self.batch_size, convert_to_numpy=True))

    def encode_docs(self, titles: list[str], abstracts: list[str]) -> np.ndarray:
        texts = [f"{t}. {a}" for t, a in zip(titles, abstracts)]
        return _l2(self.model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True))


class LSAEncoder:
    """TF-IDF + SVD. A lexical-semantic baseline that runs anywhere; not a substitute for MedCPT."""
    name = "lsa"

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.vectorizer = self.svd = None

    def fit(self, texts: list[str]) -> "LSAEncoder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        from src.index.bm25 import tokenize

        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                          sublinear_tf=True, min_df=1)
        x = self.vectorizer.fit_transform(texts)
        dim = max(2, min(self.dim, x.shape[0] - 1, x.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=dim, random_state=0).fit(x)
        return self

    def _enc(self, texts: list[str]) -> np.ndarray:
        return _l2(self.svd.transform(self.vectorizer.transform(texts)))

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self._enc(queries)

    def encode_docs(self, titles: list[str], abstracts: list[str]) -> np.ndarray:
        return self._enc([f"{t}. {a}" for t, a in zip(titles, abstracts)])

    def save(self, path: Path) -> None:
        import joblib

        joblib.dump({"vectorizer": self.vectorizer, "svd": self.svd, "dim": self.dim}, path)

    @classmethod
    def load(cls, path: Path) -> "LSAEncoder":
        import joblib

        d = joblib.load(path)
        enc = cls(d["dim"])
        enc.vectorizer, enc.svd = d["vectorizer"], d["svd"]
        return enc


def get_encoder(name: str, s: Settings, index_dir: Path | None = None, device: str | None = None,
                batch_size: int = 32) -> Encoder:
    if name == "medcpt":
        return MedCPTEncoder(s.medcpt_query_model, s.medcpt_article_model, device, batch_size)
    if name == "sbert":
        return SBERTEncoder(s.sbert_model, device, batch_size)
    if name == "lsa":
        p = (index_dir or s.index_dir / "dense") / "lsa.joblib"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: build the dense index with --encoder lsa first")
        return LSAEncoder.load(p)
    raise ValueError(f"unknown encoder {name!r}")


# ---------------------------------------------------------------- index

class DenseIndex:
    def __init__(self, embeddings: np.ndarray, ids: list[str], encoder_name: str = ""):
        self.ids = list(ids)
        self.encoder_name = encoder_name
        self.emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        try:
            import faiss

            self._faiss = faiss.IndexFlatIP(self.emb.shape[1])
            self._faiss.add(self.emb)
        except ImportError:
            self._faiss = None

    def __len__(self) -> int:
        return len(self.ids)

    def search_vectors(self, q: np.ndarray, k: int) -> list[list[tuple[str, float]]]:
        q = np.ascontiguousarray(q, dtype=np.float32)
        k = min(k, len(self.ids))
        if self._faiss is not None:
            scores, idx = self._faiss.search(q, k)
        else:
            sims = q @ self.emb.T
            idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            part = np.take_along_axis(sims, idx, axis=1)
            order = np.argsort(-part, axis=1)
            idx = np.take_along_axis(idx, order, axis=1)
            scores = np.take_along_axis(part, order, axis=1)
        return [[(self.ids[j], float(sc)) for j, sc in zip(ri, rs) if j >= 0] for ri, rs in zip(idx, scores)]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.emb)
        (directory / "meta.json").write_text(json.dumps({"encoder": self.encoder_name, "ids": self.ids,
                                                         "dim": int(self.emb.shape[1])}))

    @classmethod
    def load(cls, directory: Path) -> "DenseIndex":
        meta = json.loads((directory / "meta.json").read_text())
        return cls(np.load(directory / "embeddings.npy"), meta["ids"], meta["encoder"])


def build(corpus_path: Path, out_dir: Path, encoder_name: str, s: Settings, device: str | None = None,
          batch_size: int = 32, chunk: int = 2048) -> DenseIndex:
    df = read_table(corpus_path)
    titles, abstracts = df["title"].tolist(), df["abstract"].tolist()
    out_dir.mkdir(parents=True, exist_ok=True)
    if encoder_name == "lsa":
        enc = LSAEncoder().fit(df["text"].tolist())
        enc.save(out_dir / "lsa.joblib")
    else:
        enc = get_encoder(encoder_name, s, out_dir, device, batch_size)
    chunk_dir = out_dir / f"chunks_{encoder_name}"
    chunk_dir.mkdir(exist_ok=True)
    parts, t0 = [], time.time()
    for start in range(0, len(df), chunk):
        f = chunk_dir / f"{start:07d}.npy"
        if f.exists():  # resume after an interrupted run (e.g. a Colab disconnect)
            parts.append(np.load(f))
            continue
        v = enc.encode_docs(titles[start:start + chunk], abstracts[start:start + chunk])
        np.save(f, v)
        parts.append(v)
        done = start + len(v)
        log.info("encoded %d / %d (%.1f docs/s)", done, len(df), done / max(time.time() - t0, 1e-6))
    idx = DenseIndex(np.vstack(parts), df["pmid"].astype(str).tolist(), encoder_name)
    idx.save(out_dir)
    log.info("dense index (%s, dim=%d) saved -> %s", encoder_name, idx.emb.shape[1], out_dir)
    return idx


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=s.corpus_path)
    ap.add_argument("--out", type=Path, default=s.index_dir / "dense")
    ap.add_argument("--encoder", default=s.encoder, choices=["medcpt", "sbert", "lsa"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)
    build(args.corpus, args.out, args.encoder, s, args.device, args.batch_size)


if __name__ == "__main__":
    main()
