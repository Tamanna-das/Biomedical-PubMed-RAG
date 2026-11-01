"""Small I/O helpers: parquet tables (with a JSONL fallback when pyarrow is missing),
JSONL read/write and logging setup."""
from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

log = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


def _fallback(path: Path) -> Path:
    return path.with_suffix(".jsonl.gz")


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _has_pyarrow():
        df.to_parquet(path, index=False)
        return path
    fb = _fallback(path)
    log.warning("pyarrow not installed; writing %s instead of parquet", fb)
    with gzip.open(fb, "wt", encoding="utf-8") as f:
        df.to_json(f, orient="records", lines=True, force_ascii=False)
    return fb


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    fb = _fallback(path)
    if fb.exists():
        with gzip.open(fb, "rt", encoding="utf-8") as f:
            return pd.read_json(f, orient="records", lines=True, dtype={"pmid": str})
    raise FileNotFoundError(f"No table at {path} (or {fb}). Run the ingest step first.")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(rows: Iterable[dict], path: str | Path, append: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
