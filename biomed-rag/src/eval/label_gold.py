"""Build gold PMIDs for eval/questions.jsonl by pooled relevance judging.

Writing questions and gold labels from abstracts you just read biases the set toward BM25,
and a single gold PMID per question undercounts recall. Pooling fixes both: every method's
top-10 for each question goes into one sheet and you judge each candidate once.

    python -m src.eval.label_gold pool      # -> eval/pool_to_judge.csv (fill the `relevant` column: 1/0)
    python -m src.eval.label_gold import    # -> eval/gold_pmids.jsonl

    python -m src.eval.label_gold prelabel  # optional: LLM drafts a 1/0 for every unlabelled row

Pre-labelling fills `relevant` with the LLM's guess and `llm_reason` with why; you then
review every row and set `checked` = y. Report it as "LLM-pre-labelled, human-verified".
Re-running `pool` (e.g. after adding questions or the reranker) keeps labels you already made.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import get_settings
from src.io_utils import read_jsonl, setup_logging, write_jsonl
from src.retrieve.hybrid import Retriever

YES = {"1", "y", "yes", "true", "t", "relevant"}
KEEP = ("relevant", "llm_reason", "checked")

JUDGE_SYSTEM = ("You judge search results for a biomedical literature QA system. Reply with 1 if the abstract "
                "contains information that directly helps answer the question, otherwise 0. Format: the digit, "
                "then ' - ', then a reason of at most 15 words.")


def prelabel(df: pd.DataFrame, llm, log_every: int = 25) -> pd.DataFrame:
    """Fill empty `relevant` cells with an LLM judgment. Rows already labelled are untouched."""
    df = df.copy()
    todo = df.index[df["relevant"].astype(str).str.strip() == ""]
    for n, i in enumerate(todo, 1):
        r = df.loc[i]
        msgs = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": f"Question: {r['question']}\n\nTitle: {r['title']}\nAbstract: {r['abstract']}"}]
        try:
            text = llm.generate(msgs).text.strip()
        except Exception as e:  # rate limit etc. -> leave for the human
            print(f"row {i}: judge failed ({e}); left blank")
            continue
        digit = next((c for c in text if c in "01"), "")
        df.at[i, "relevant"] = digit
        df.at[i, "llm_reason"] = text.split("-", 1)[-1].strip()[:200] if digit else text[:200]
        if n % log_every == 0:
            print(f"pre-labelled {n} / {len(todo)}")
    return df


def pool(retriever: Retriever, questions: list[dict], depth: int = 10, rerank: bool = False) -> pd.DataFrame:
    rows = []
    for q in questions:
        if q.get("answerable", True) is False:
            continue
        found: dict[str, list[str]] = {}
        runs = [(m, False) for m in retriever.methods] + ([("hybrid", True)] if rerank else [])
        for m, rr in runs:
            for h in retriever.search(q["question"], m, depth, rr):
                found.setdefault(h.pmid, []).append(f"{m}{'+rr' if rr else ''}:{h.rank}")
        for pmid, how in found.items():
            d = retriever.docs[pmid]
            rows.append({"qid": q["qid"], "question": q["question"], "pmid": pmid, "found_by": " ".join(how),
                         "title": d["title"], "abstract": d["abstract"][:700], "relevant": "", "llm_reason": "",
                         "checked": ""})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["pool", "prelabel", "import"])
    ap.add_argument("--backend", default=None, help="LLM backend for prelabel (default: LLM_BACKEND)")
    ap.add_argument("--sheet", type=Path, default=s.eval_dir / "pool_to_judge.csv")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--rerank", action="store_true")
    args = ap.parse_args(argv)
    qpath, gpath = s.eval_dir / "questions.jsonl", s.eval_dir / "gold_pmids.jsonl"

    if args.action == "pool":
        retriever = Retriever.from_disk(s, use_reranker=args.rerank)
        df = pool(retriever, read_jsonl(qpath), args.depth, args.rerank)
        if args.sheet.exists():  # keep previous judgments
            old = pd.read_csv(args.sheet, dtype=str).fillna("")
            for col in KEEP:
                if col in old:
                    prev = {(q, p): v for q, p, v in zip(old["qid"], old["pmid"], old[col]) if v}
                    df[col] = [prev.get((q, p), "") for q, p in zip(df["qid"], df["pmid"])]
        df.to_csv(args.sheet, index=False)
        todo = (df["relevant"] == "").sum()
        print(f"wrote {len(df)} candidates for {df['qid'].nunique()} questions -> {args.sheet} ({todo} to judge)")
        return

    if args.action == "prelabel":
        from src.generate.llm import get_llm

        df = pd.read_csv(args.sheet, dtype=str).fillna("")
        for col in KEEP:
            if col not in df:
                df[col] = ""
        df = prelabel(df, get_llm(s, args.backend))
        df.to_csv(args.sheet, index=False)
        print(f"pre-labelled sheet -> {args.sheet}. Review every row, fix `relevant`, set `checked` = y.")
        return

    df = pd.read_csv(args.sheet, dtype=str).fillna("")
    gold = {}
    for r in df.itertuples():
        gold.setdefault(r.qid, [])
        if r.relevant.strip().lower() in YES:
            gold[r.qid].append(r.pmid)
    write_jsonl([{"qid": q, "pmids": p} for q, p in gold.items() if p], gpath)
    empty = [q for q, p in gold.items() if not p]
    unjudged = (df["relevant"].str.strip() == "").sum()
    print(f"gold for {sum(1 for p in gold.values() if p)} questions -> {gpath}")
    if empty:
        print(f"no relevant PMID found for: {', '.join(empty)} -> rewrite them or mark answerable=false")
    if unjudged:
        print(f"warning: {unjudged} candidates still unjudged (treated as not relevant)")
    if "checked" in df and (df["llm_reason"] != "").any():
        unchecked = ((df["llm_reason"] != "") & (df["checked"].str.strip() == "")).sum()
        if unchecked:
            print(f"warning: {unchecked} LLM-pre-labelled rows not yet human-checked (`checked` column empty)")


if __name__ == "__main__":
    main()
