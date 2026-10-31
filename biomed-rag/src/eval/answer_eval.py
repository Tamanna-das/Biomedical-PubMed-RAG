"""Answer-level evaluation: grounded prompt vs plain-prompt baseline, same retrieval.

    python -m src.eval.answer_eval run            # answers + automatic metrics + labelling sheet
    (label results/faithfulness_sheet.csv: label = supported | partial | unsupported)
    python -m src.eval.answer_eval summarize      # faithfulness table + failure cases

Automatic metrics (every question):
  answerable questions   false-abstention rate, % answers with a citation, % with an invalid
                         (not-retrieved) citation, uncited-sentence rate, citation precision vs
                         gold (a lower bound: gold sets are incomplete)
  unanswerable questions correct-abstention rate
  cost                   latency mean / p95, tokens and USD per query
Manual (sample, 15 per mode by default): each answer judged against its cited abstracts. This is the faithfulness
number for the README; LLM-as-judge with a small local model is too noisy to trust alone.
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_settings
from src.io_utils import read_jsonl, setup_logging, write_jsonl
from src.pipeline import RAGPipeline

log = logging.getLogger("answer_eval")
MODES = ("grounded", "plain")


def auto_metrics(records: list[dict]) -> dict:
    ans = [r for r in records if r["answerable"]]
    una = [r for r in records if not r["answerable"]]
    answered = [r for r in ans if not r["abstained"]]
    lat = [r["latency_s"] for r in records if r["abstain_reason"] != "retrieval_gate"] or [0.0]
    prec = [len(set(r["citations"]) & set(r["gold"])) / len(r["citations"])
            for r in answered if r["citations"] and r["gold"]]
    sents = sum(r["n_sentences"] for r in answered)
    return {
        "n_answerable": len(ans),
        "false_abstention": np.mean([r["abstained"] for r in ans]) if ans else float("nan"),
        "answers_with_citation": np.mean([bool(r["citations"]) for r in answered]) if answered else float("nan"),
        "answers_with_invalid_citation": np.mean([bool(r["invalid_citations"]) for r in answered]) if answered else float("nan"),
        "uncited_sentence_rate": (sum(r["uncited_sentences"] for r in answered) / sents) if sents else float("nan"),
        "citation_precision_vs_gold": np.mean(prec) if prec else float("nan"),
        "n_unanswerable": len(una),
        "correct_abstention": np.mean([r["abstained"] for r in una]) if una else float("nan"),
        "latency_mean_s": float(np.mean(lat)),
        "latency_p95_s": float(np.quantile(lat, 0.95)),
        "tokens_per_query": float(np.mean([r["input_tokens"] + r["output_tokens"] for r in records])) if records else 0,
        "usd_per_query": float(np.mean([r["cost_usd"] for r in records])) if records else 0,
    }


def metrics_table(by_mode: dict[str, dict]) -> str:
    keys = list(next(iter(by_mode.values())).keys())
    lines = ["| metric | " + " | ".join(by_mode) + " |", "|---|" + "---|" * len(by_mode)]
    for k in keys:
        cells = []
        for m in by_mode:
            v = by_mode[m][k]
            cells.append(f"{v:.6f}" if k == "usd_per_query" else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run(args, s) -> None:
    pipe = RAGPipeline.from_settings(s, use_reranker=args.rerank, backend=args.backend)
    gold = {g["qid"]: g["pmids"] for g in read_jsonl(s.eval_dir / "gold_pmids.jsonl")}
    questions = read_jsonl(s.eval_dir / "questions.jsonl")[: args.limit]
    by_mode, all_recs = {}, []
    for mode in MODES:
        recs = []
        for i, q in enumerate(questions, 1):
            a = pipe.ask(q["question"], args.method, args.k, args.rerank, mode).to_dict()
            a.pop("hits")
            recs.append({"qid": q["qid"], "answerable": q.get("answerable", True), "gold": gold.get(q["qid"], []), **a})
            if i % 10 == 0:
                log.info("[%s] %d / %d", mode, i, len(questions))
        write_jsonl(recs, s.results_dir / f"answers_{mode}.jsonl")
        by_mode[mode] = auto_metrics(recs)
        all_recs += recs

    md = [f"**Answer evaluation** — retrieval `{args.method}` top-{args.k or s.top_k}, "
          f"LLM `{pipe.llm.model}`, {len(questions)} questions\n", metrics_table(by_mode), ""]
    (s.results_dir / "answer_eval.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # labelling sheet: a random sample of non-abstained answers per mode, plus all unanswerable ones
    rng = random.Random(0)
    sheet = []
    for mode in MODES:
        pool = [r for r in all_recs if r["mode"] == mode and not r["abstained"]]
        for r in rng.sample(pool, min(args.sample, len(pool))):
            sheet.append({"qid": r["qid"], "mode": mode, "question": r["question"], "answer": r["answer"],
                          "cited": " ".join(r["citations"]), "context_pmids": " ".join(r["context_pmids"]),
                          "label": "", "notes": ""})
    path = s.results_dir / "faithfulness_sheet.csv"
    if path.exists() and not args.overwrite_sheet:
        path = path.with_name("faithfulness_sheet.new.csv")
    pd.DataFrame(sheet).to_csv(path, index=False)
    print(f"labelling sheet ({len(sheet)} rows) -> {path}")


def summarize(args, s) -> None:
    df = pd.read_csv(s.results_dir / "faithfulness_sheet.csv", dtype=str).fillna("")
    df["label"] = df["label"].str.strip().str.lower()
    lab = df[df["label"].isin(["supported", "partial", "unsupported"])]
    if lab.empty:
        raise SystemExit("No labels yet: fill the `label` column with supported / partial / unsupported.")
    tab = pd.crosstab(lab["mode"], lab["label"], normalize="index").round(3)
    counts = lab.groupby("mode").size().rename("n_labelled")
    out = tab.join(counts)
    md = ["**Faithfulness (manual labels; answer judged against its cited abstracts)**", "",
          out.to_markdown() if hasattr(out, "to_markdown") else out.to_string(), ""]
    fails = lab[lab["label"] != "supported"]
    fl = ["# Failure cases", ""]
    for r in fails.itertuples():
        fl += [f"## {r.qid} ({r.mode}, {r.label})", f"**Q:** {r.question}", "", f"**A:** {r.answer}", "",
               f"**Cited:** {r.cited or '—'}  ", f"**Notes:** {r.notes or '—'}", ""]
    (s.results_dir / "faithfulness.md").write_text("\n".join(md) + "\n")
    (s.results_dir / "failures.md").write_text("\n".join(fl) + "\n")
    print("\n".join(md))
    print(f"{len(fails)} failure cases -> {s.results_dir / 'failures.md'}")


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "summarize"])
    ap.add_argument("--method", default="hybrid")
    ap.add_argument("-k", type=int, default=None)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", type=int, default=15, help="answers per mode in the labelling sheet")
    ap.add_argument("--overwrite-sheet", action="store_true")
    args = ap.parse_args(argv)
    s.results_dir.mkdir(parents=True, exist_ok=True)
    run(args, s) if args.action == "run" else summarize(args, s)


if __name__ == "__main__":
    main()
