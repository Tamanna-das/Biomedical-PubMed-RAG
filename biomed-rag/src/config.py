"""Central configuration. Every value can be overridden with an environment variable
(or a .env file in the repo root)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:  # optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


@dataclass(frozen=True)
class Settings:
    # paths
    data_dir: Path
    raw_path: Path
    corpus_path: Path
    index_dir: Path
    results_dir: Path
    eval_dir: Path
    # PubMed
    ncbi_email: str
    ncbi_api_key: str
    # retrieval
    encoder: str  # medcpt | sbert | lsa
    sbert_model: str
    medcpt_query_model: str
    medcpt_article_model: str
    reranker_model: str
    rrf_k: int
    candidate_depth: int
    top_k: int
    abstain_threshold: float  # min dense cosine of best hit; below -> abstain
    # generation
    llm_backend: str  # ollama | openai | extractive
    ollama_url: str
    ollama_model: str
    openai_base_url: str
    openai_api_key: str
    openai_model: str
    price_in_per_m: float  # USD per 1M input tokens
    price_out_per_m: float  # USD per 1M output tokens
    max_context_chars: int


def get_settings() -> Settings:
    data_dir = Path(_env("BIOMED_DATA_DIR", str(ROOT / "data")))
    index_dir = Path(_env("BIOMED_INDEX_DIR", str(data_dir / "index")))
    return Settings(
        data_dir=data_dir,
        raw_path=data_dir / "raw" / "records.jsonl",
        corpus_path=Path(_env("BIOMED_CORPUS", str(data_dir / "processed" / "corpus.parquet"))),
        index_dir=index_dir,
        results_dir=Path(_env("BIOMED_RESULTS_DIR", str(ROOT / "results"))),
        eval_dir=Path(_env("BIOMED_EVAL_DIR", str(ROOT / "eval"))),
        ncbi_email=_env("NCBI_EMAIL", ""),
        ncbi_api_key=_env("NCBI_API_KEY", ""),
        encoder=_env("ENCODER", "medcpt"),
        sbert_model=_env("SBERT_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO"),
        medcpt_query_model=_env("MEDCPT_QUERY_MODEL", "ncbi/MedCPT-Query-Encoder"),
        medcpt_article_model=_env("MEDCPT_ARTICLE_MODEL", "ncbi/MedCPT-Article-Encoder"),
        reranker_model=_env("RERANKER_MODEL", "ncbi/MedCPT-Cross-Encoder"),
        rrf_k=_int("RRF_K", 60),
        candidate_depth=_int("CANDIDATE_DEPTH", 100),
        top_k=_int("TOP_K", 5),
        abstain_threshold=_float("ABSTAIN_THRESHOLD", 0.0),
        llm_backend=_env("LLM_BACKEND", "ollama"),
        ollama_url=_env("OLLAMA_URL", "http://localhost:11434"),
        ollama_model=_env("OLLAMA_MODEL", "llama3.1:8b"),
        openai_base_url=_env("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
        openai_api_key=_env("OPENAI_API_KEY", ""),
        openai_model=_env("OPENAI_MODEL", "llama-3.1-8b-instant"),
        price_in_per_m=_float("PRICE_IN_PER_M", 0.0),
        price_out_per_m=_float("PRICE_OUT_PER_M", 0.0),
        max_context_chars=_int("MAX_CONTEXT_CHARS", 12000),
    )
