import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def demo_env(tmp_path_factory):
    """Fixture corpus indexed with BM25 + LSA in a temp dir; returns Settings."""
    import os

    from src.config import get_settings
    from src.index import bm25, dense
    from src.ingest.clean import clean_records
    from src.io_utils import read_jsonl, write_table

    d = tmp_path_factory.mktemp("data")
    os.environ.update({"BIOMED_DATA_DIR": str(d), "ENCODER": "lsa", "LLM_BACKEND": "extractive",
                       "BIOMED_RESULTS_DIR": str(d / "results"), "ABSTAIN_THRESHOLD": "0.0"})
    s = get_settings()
    df = clean_records(read_jsonl(FIXTURES / "sample_records.jsonl"), min_chars=100)
    write_table(df, s.corpus_path)
    bm25.build(s.corpus_path, s.index_dir / "bm25")
    dense.build(s.corpus_path, s.index_dir / "dense", "lsa", s)
    return s


@pytest.fixture(scope="session")
def retriever(demo_env):
    from src.retrieve.hybrid import Retriever

    return Retriever.from_disk(demo_env)
