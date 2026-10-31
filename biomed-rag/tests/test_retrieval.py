import math

import numpy as np
import pytest

from src.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from src.index.bm25 import BM25Index, tokenize
from src.index.dense import DenseIndex
from src.retrieve.hybrid import rrf

DOCS = ["colistin resistance mcr gene plasmid", "carbapenem resistance ndm enzyme",
        "efflux pump pseudomonas", "colistin colistin heteroresistance klebsiella"]


def reference_bm25(query, docs, k1=1.5, b=0.75):
    toks = [tokenize(d) for d in docs]
    n, avg = len(toks), sum(map(len, toks)) / len(toks)
    out = []
    for t in toks:
        s = 0.0
        for q in tokenize(query):
            df = sum(q in d for d in toks)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            tf = t.count(q)
            s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(t) / avg))
        out.append(s)
    return np.array(out)


def test_bm25_matches_reference_formula(tmp_path):
    idx = BM25Index().fit(DOCS, ["a", "b", "c", "d"])
    for q in ["colistin resistance", "ndm carbapenem enzyme", "unknownword", "efflux efflux"]:
        np.testing.assert_allclose(idx.scores(q), reference_bm25(q, DOCS), rtol=1e-5)
    idx.save(tmp_path)
    loaded = BM25Index.load(tmp_path)
    assert loaded.search("colistin", 2) == idx.search("colistin", 2)
    assert idx.search("unknownword", 3) == []


def test_tokenize_light_stemming():
    assert tokenize("The Enzymes and bacteria's pumps") == ["enzyme", "bacteria", "pump"]


def test_rrf():
    fused = dict(rrf([["a", "b", "c"], ["b", "a", "d"]], k=60))
    assert fused["a"] == fused["b"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["c"] == pytest.approx(1 / 63) and fused["d"] == pytest.approx(1 / 63)


def test_dense_numpy_exact_search():
    emb = np.eye(4, dtype=np.float32)
    idx = DenseIndex(emb, ["a", "b", "c", "d"], "test")
    res = idx.search_vectors(np.array([[0.1, 0.9, 0.0, 0.4]], dtype=np.float32), 2)[0]
    assert [r[0] for r in res] == ["b", "d"]


def test_metrics():
    ranked, gold = ["x", "a", "y", "b"], {"a", "b"}
    assert recall_at_k(ranked, gold, 2) == 0.5 and recall_at_k(ranked, gold, 4) == 1.0
    assert mrr_at_k(ranked, gold) == 0.5
    assert ndcg_at_k(["a", "b"], gold) == pytest.approx(1.0)


@pytest.mark.parametrize("method", ["bm25", "dense", "hybrid"])
def test_retriever_finds_obvious_doc(retriever, method):
    hits = retriever.search("mecA PBP2a methicillin-resistant Staphylococcus aureus", method, k=3)
    assert hits[0].pmid == "99000007"
    assert all(h.dense_score is not None for h in hits)
    assert [h.rank for h in hits] == [1, 2, 3]


def test_hybrid_records_component_ranks(retriever):
    h = retriever.search("vancomycin-resistant enterococci vanA", "hybrid", k=1)[0]
    assert h.pmid == "99000008" and h.bm25_rank == 1 and h.dense_rank is not None


def test_unknown_method_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.search("x", "splade", 3)
