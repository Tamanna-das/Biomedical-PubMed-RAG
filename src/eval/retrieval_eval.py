"""Compare BM25 vs dense vs hybrid (vs hybrid+rerank) on recall@k, MRR@10, nDCG@10.

    python -m src.eval.retrieval_eval                    # your AMR set (eval/questions.jsonl + gold)
    python -m src.eval.retrieval_eval --dataset pubmedqa # PubMedQA benchmark index
    python -m src.eval.retrieval_eval --rerank           # also score hybrid + cross-encoder

Writes results/retrieval_<dataset>.md (the README table), .csv, and per-query rankings
(results/retrieval_<dataset>_per_query.jsonl) for failure analysis. The table is written
whichever method wins.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from src.config import get_settings
from src.eval.metrics import bootstrap_ci, hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from src.io_utils import read_jsonl, setup_logging, write_jsonl
from src.retrieve.hybrid import Retriever

log = logging.getLogger("retrieval_eval")
KS = (1, 5, 10, 20)


def load_eval_set(questions_path: Path, gold_path: Path) -> list[dict]:
    gold = {g["qid"]: set(map(str, g["pmids"])) for g in read_jsonl(gold_path)}
    items = []
    for q in read_jsonl(questions_path):
        g = gold.get(q["qid"], set())
        if g:  # unanswerable / unlabelled questions are excluded from retrieval metrics
            items.append({**q, "gold": g})
    return items


def evaluate(retriever: Retriever, items: list[dict], methods: list[str], rerank: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    runs = [(m, False) for m in methods] + ([("hybrid", True)] if rerank else [])
    rows, per_query = [], []
    for method, rr in runs:
        name = method + ("+rerank" if rr else "")
        per = {f"recall@{k}": [] for k in KS} | {"hit@5": [], "mrr@10": [], "ndcg@10": []}
        t0 = time.time()
        for it in items:
            ranked = retriever.ranked_ids(it["question"], method, max(KS), rr)
            g = it["gold"]
            for k in KS:
                per[f"recall@{k}"].append(recall_at_k(ranked, g, k))
            per["hit@5"].append(hit_at_k(ranked, g, 5))
            per["mrr@10"].append(mrr_at_k(ranked, g, 10))
            per["ndcg@10"].append(ndcg_at_k(ranked, g, 10))
            per_query.append({"method": name, "qid": it["qid"], "question": it["question"], "gold": sorted(g),
                              "top10": ranked[:10], "recall@5": per["recall@5"][-1], "mrr@10": per["mrr@10"][-1]})
        ms = (time.time() - t0) / max(len(items), 1) * 1000
        row = {"method": name, **{m: sum(v) / len(v) for m, v in per.items()}, "ms/query": ms}
        lo, hi = bootstrap_ci(per["recall@5"])
        row["recall@5 95% CI"] = f"[{lo:.3f}, {hi:.3f}]"
        rows.append(row)
        log.info("%-14s recall@5=%.3f mrr@10=%.3f (%.0f ms/q)", name, row["recall@5"], row["mrr@10"], ms)
    return pd.DataFrame(rows), per_query


def to_markdown(df: pd.DataFrame, dataset: str, n: int) -> str:
    cols = ["method", "recall@1", "recall@5", "recall@10", "recall@20", "mrr@10", "ndcg@10", "recall@5 95% CI", "ms/query"]
    lines = [f"**Retrieval on `{dataset}` ({n} questions with gold PMIDs)**", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    best = {c: df[c].max() for c in cols[1:7]}
    for _, r in df.iterrows():
        cells = [r["method"]]
        for c in cols[1:7]:
            v = f"{r[c]:.3f}"
            cells.append(f"**{v}**" if r[c] == best[c] else v)
        cells += [r["recall@5 95% CI"], f"{r['ms/query']:.0f}"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Bold = best in column. CI = bootstrap over questions; overlapping CIs mean the difference "
              "is not reliable at this sample size."]
    return "\n".join(lines) + "\n"


def paths_for(dataset: str, s) -> tuple[Path, Path, Path, Path]:
    if dataset == "own":
        return s.eval_dir / "questions.jsonl", s.eval_dir / "gold_pmids.jsonl", s.corpus_path, s.index_dir
    if dataset == "pubmedqa":
        d = s.data_dir / "pubmedqa"
        return s.eval_dir / "pubmedqa" / "questions.jsonl", s.eval_dir / "pubmedqa" / "gold_pmids.jsonl", \
            d / "corpus.parquet", d / "index"
    raise ValueError(dataset)


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="own", choices=["own", "pubmedqa"])
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    qpath, gpath, corpus, index_dir = paths_for(args.dataset, s)
    items = load_eval_set(qpath, gpath)[: args.limit]
    if not items:
        raise SystemExit(f"No questions with gold PMIDs in {gpath}. For your own set, run "
                         "`python -m src.eval.label_gold pool` and label the pool first.")
    retriever = Retriever.from_disk(s, corpus_path=corpus, index_dir=index_dir, use_reranker=args.rerank)
    df, per_query = evaluate(retriever, items, list(retriever.methods), args.rerank)

    s.results_dir.mkdir(parents=True, exist_ok=True)
    stem = s.results_dir / f"retrieval_{args.dataset}"
    md = to_markdown(df, args.dataset, len(items))
    Path(f"{stem}.md").write_text(md)
    df.to_csv(f"{stem}.csv", index=False)
    write_jsonl(per_query, f"{stem}_per_query.jsonl")
    print(md)


if __name__ == "__main__":
    main()
