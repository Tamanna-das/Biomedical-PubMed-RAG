"""Clean raw PubMed records into the corpus table used by every index.

    python -m src.ingest.clean            # data/raw/records.jsonl -> data/processed/corpus.parquet
"""
from __future__ import annotations

import argparse
import logging
import unicodedata
from pathlib import Path

import pandas as pd

from src.config import get_settings
from src.io_utils import read_jsonl, setup_logging, write_table

log = logging.getLogger("clean")

EXCLUDED_TYPES = {"Retracted Publication", "Retraction of Publication", "Erratum", "Published Erratum",
                  "Comment", "Editorial", "Letter", "News"}
MIN_ABSTRACT_CHARS = 200


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return " ".join(text.split())


def clean_records(records: list[dict], min_chars: int = MIN_ABSTRACT_CHARS) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["pmid", "title", "abstract", "journal", "year", "doi", "mesh", "text"])
    n0 = len(df)
    for col in ("title", "abstract", "journal", "doi"):
        df[col] = df.get(col, "").fillna("").map(normalize)
    for col in ("mesh", "pub_types"):
        if col not in df:
            df[col] = [[] for _ in range(len(df))]
        df[col] = df[col].map(lambda v: list(v) if isinstance(v, (list, tuple)) else [])
    df["pmid"] = df["pmid"].astype(str)
    df = df.drop_duplicates("pmid")
    df = df[df["abstract"].str.len() >= min_chars]
    df = df[~df["pub_types"].map(lambda t: bool(EXCLUDED_TYPES.intersection(t)))]
    df["year"] = pd.to_numeric(df.get("year"), errors="coerce").astype("Int64")
    df["title"] = df["title"].str.rstrip(".")
    df["text"] = df["title"] + ". " + df["abstract"]
    df = df.drop(columns=["pub_types"]).reset_index(drop=True)
    log.info("kept %d / %d records", len(df), n0)
    return df


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=s.raw_path)
    ap.add_argument("--out", type=Path, default=s.corpus_path)
    ap.add_argument("--min-chars", type=int, default=MIN_ABSTRACT_CHARS)
    args = ap.parse_args(argv)
    df = clean_records(read_jsonl(args.raw), args.min_chars)
    path = write_table(df, args.out)
    yrs = df["year"].dropna()
    log.info("corpus: %d abstracts, years %s-%s, %d journals -> %s", len(df),
             yrs.min() if len(yrs) else "?", yrs.max() if len(yrs) else "?", df["journal"].nunique(), path)


if __name__ == "__main__":
    main()
