"""PubMedQA (PQA-L, 1,000 expert-labelled questions) as a public retrieval benchmark.

PubMedQA questions are not about AMR and their source abstracts are mostly not in the AMR
corpus, so they get their own index: the 1,000 PubMedQA contexts, optionally padded with
distractor abstracts from your AMR corpus to make retrieval harder. Gold = the question's
own PMID. Note the contexts exclude each abstract's conclusion section.

    python -m src.eval.pubmedqa --encoder medcpt --distractors 20000
    python -m src.eval.retrieval_eval --dataset pubmedqa
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import requests

from src.config import get_settings
from src.index import bm25 as bm25_mod
from src.index import dense as dense_mod
from src.io_utils import read_table, setup_logging, write_jsonl, write_table

log = logging.getLogger("pubmedqa")
URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"


def load_pqal(path: Path) -> dict:
    if not path.exists():
        log.info("downloading %s", URL)
        r = requests.get(URL, timeout=120)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    return json.loads(path.read_text())


def build_benchmark(pqal: dict, distractors: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    docs, questions, gold = [], [], []
    for pmid, ex in pqal.items():
        docs.append({"pmid": str(pmid), "title": "", "abstract": " ".join(ex["CONTEXTS"]),
                     "journal": "", "year": int(ex["YEAR"]) if str(ex.get("YEAR", "")).isdigit() else None})
        questions.append({"qid": f"pqa_{pmid}", "question": ex["QUESTION"], "answerable": True,
                          "final_decision": ex.get("final_decision")})
        gold.append({"qid": f"pqa_{pmid}", "pmids": [str(pmid)]})
    df = pd.DataFrame(docs)
    if distractors is not None and len(distractors):
        d = distractors[~distractors["pmid"].astype(str).isin(set(df["pmid"]))]
        df = pd.concat([df, d[["pmid", "title", "abstract", "journal", "year"]]], ignore_index=True)
    df["text"] = (df["title"].fillna("") + ". " + df["abstract"]).str.lstrip(". ")
    return df, questions, gold


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=s.encoder, choices=["medcpt", "sbert", "lsa"])
    ap.add_argument("--distractors", type=int, default=0, help="AMR abstracts to add as distractors")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    root = s.data_dir / "pubmedqa"
    pqal = load_pqal(root / "ori_pqal.json")
    distr = read_table(s.corpus_path).sample(frac=1, random_state=0).head(args.distractors) if args.distractors else None
    df, questions, gold = build_benchmark(pqal, distr)
    write_table(df, root / "corpus.parquet")
    write_jsonl(questions, s.eval_dir / "pubmedqa" / "questions.jsonl")
    write_jsonl(gold, s.eval_dir / "pubmedqa" / "gold_pmids.jsonl")
    log.info("PubMedQA benchmark: %d docs (%d distractors), %d questions", len(df), len(df) - len(pqal), len(questions))
    bm25_mod.build(root / "corpus.parquet", root / "index" / "bm25")
    dense_mod.build(root / "corpus.parquet", root / "index" / "dense", args.encoder, s, args.device)


if __name__ == "__main__":
    main()
