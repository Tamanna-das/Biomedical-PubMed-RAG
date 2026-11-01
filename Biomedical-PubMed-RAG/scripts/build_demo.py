"""Build a tiny offline demo index (no internet, no GPU, no LLM) from the synthetic test
fixture, then answer one question. Useful to check an install end to end.

    python scripts/build_demo.py
    BIOMED_DATA_DIR=data/demo ENCODER=lsa LLM_BACKEND=extractive uvicorn api.main:app

The 16 fixture abstracts are synthetic (PMIDs 990000xx are not real PubMed records).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("BIOMED_DATA_DIR", str(ROOT / "data" / "demo"))
os.environ.setdefault("ENCODER", "lsa")
os.environ.setdefault("LLM_BACKEND", "extractive")

from src.config import get_settings  # noqa: E402
from src.index import bm25, dense  # noqa: E402
from src.ingest.clean import clean_records  # noqa: E402
from src.io_utils import read_jsonl, setup_logging, write_table  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402


def main() -> None:
    setup_logging()
    s = get_settings()
    df = clean_records(read_jsonl(ROOT / "tests" / "fixtures" / "sample_records.jsonl"), min_chars=100)
    write_table(df, s.corpus_path)
    bm25.build(s.corpus_path, s.index_dir / "bm25")
    dense.build(s.corpus_path, s.index_dir / "dense", "lsa", s)
    p = RAGPipeline.from_settings(s)
    for q in ["How does the mecA gene make Staphylococcus aureus resistant to methicillin?",
              "What dose of amoxicillin should I give my 3-year-old?"]:
        a = p.ask(q).to_dict()
        print(json.dumps({k: a[k] for k in ("question", "answer", "abstained", "citations", "invalid_citations")}, indent=2))
    print(f"\nDemo index ready in {s.data_dir}. Serve it with:\n"
          f"  BIOMED_DATA_DIR={s.data_dir} ENCODER=lsa LLM_BACKEND=extractive uvicorn api.main:app")


if __name__ == "__main__":
    main()
