"""FastAPI service.

    uvicorn api.main:app --reload
    curl -X POST localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question": "How does NDM-1 confer carbapenem resistance?"}'
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline import get_pipeline

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_pipeline()  # load indexes and models once at startup
    except Exception as e:  # keep the server up so /health can report the problem
        app.state.load_error = str(e)
    yield


app = FastAPI(title="Biomedical PubMed RAG", version="1.0.0", lifespan=lifespan,
              description="Citation-grounded QA over PubMed abstracts on antimicrobial resistance.")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    method: Literal["bm25", "dense", "hybrid"] = "hybrid"
    k: int = Field(5, ge=1, le=20)
    rerank: bool = False
    mode: Literal["grounded", "plain"] = "grounded"


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    method: Literal["bm25", "dense", "hybrid"] = "hybrid"
    k: int = Field(10, ge=1, le=50)
    rerank: bool = False


@app.get("/health")
def health() -> dict:
    err = getattr(app.state, "load_error", None)
    if err:
        return {"status": "error", "detail": err}
    p = get_pipeline()
    return {"status": "ok", "documents": len(p.retriever.docs), "methods": list(p.retriever.methods),
            "encoder": p.retriever.dense.encoder_name if p.retriever.dense else None,
            "llm": p.llm.model, "reranker": p.retriever.reranker is not None}


def _pipe():
    try:
        return get_pipeline()
    except Exception as e:
        raise HTTPException(503, f"pipeline not ready: {e}") from e


@app.post("/search")
def search(req: SearchRequest) -> dict:
    p = _pipe()
    try:
        hits = p.retriever.search(req.query, req.method, req.k, req.rerank)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"query": req.query, "method": req.method, "hits": [h.to_dict() for h in hits]}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    p = _pipe()
    try:
        return p.ask(req.question, req.method, req.k, req.rerank, req.mode).to_dict()
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # LLM backend down, etc.
        raise HTTPException(502, f"generation failed: {e}") from e
