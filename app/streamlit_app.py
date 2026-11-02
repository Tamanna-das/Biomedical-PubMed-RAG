"""Streamlit UI.  streamlit run app/streamlit_app.py

Talks to the FastAPI service at API_URL (default http://localhost:8000). If API_URL is set
to "local", it loads the pipeline in-process instead (no API needed).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
ABSTAIN = "INSUFFICIENT_EVIDENCE"
EXAMPLES = [
    "Which genes confer plasmid-mediated colistin resistance?",
    "How do carbapenemases such as NDM-1 inactivate carbapenems?",
    "What role do efflux pumps play in multidrug resistance in Pseudomonas aeruginosa?",
    "Does antibiotic use in livestock contribute to resistance in human pathogens?",
]

st.set_page_config(page_title="AMR Literature Assistant", page_icon="🧫", layout="wide")


@st.cache_resource
def _local_pipeline():
    from src.pipeline import get_pipeline

    return get_pipeline()


def ask(question: str, method: str, k: int, rerank: bool, mode: str) -> dict:
    if API_URL == "local":
        return _local_pipeline().ask(question, method, k, rerank, mode).to_dict()
    r = requests.post(f"{API_URL}/ask", json={"question": question, "method": method, "k": k,
                                              "rerank": rerank, "mode": mode}, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(r.json().get("detail", r.text))
    return r.json()


def health() -> dict:
    if API_URL == "local":
        p = _local_pipeline()
        return {"status": "ok", "documents": len(p.retriever.docs), "methods": list(p.retriever.methods), "llm": p.llm.model}
    try:
        return requests.get(f"{API_URL}/health", timeout=10).json()
    except requests.RequestException as e:
        return {"status": "error", "detail": f"API unreachable at {API_URL}: {e}"}


def linkify(text: str) -> str:
    return re.sub(r"\[?PMID:?\s*(\d{5,9})\]?", r"[[PMID:\1]](https://pubmed.ncbi.nlm.nih.gov/\1/)", text)


with st.sidebar:
    st.header("Settings")
    method = st.radio("Retrieval", ["hybrid", "dense", "bm25"], help="hybrid = BM25 + dense fused with RRF")
    k = st.slider("Abstracts given to the LLM", 1, 10, 5)
    rerank = st.checkbox("Cross-encoder rerank", help="needs USE_RERANKER=1 on the server")
    mode = st.radio("Prompt", ["grounded", "plain"], help="plain = baseline without citation rules")
    st.divider()
    h = health()
    if h.get("status") == "ok":
        st.caption(f"✅ {h['documents']:,} abstracts · {', '.join(h['methods'])} · LLM: {h['llm']}")
    else:
        st.error(h.get("detail", "pipeline not ready"))

st.title("Antimicrobial resistance literature assistant")
st.caption("Answers come only from retrieved PubMed abstracts and cite their PMIDs. "
           "Research demo — not medical advice.")

q = st.text_input("Question", placeholder=EXAMPLES[0])
cols = st.columns(len(EXAMPLES))
for c, ex in zip(cols, EXAMPLES):
    if c.button(ex[:42] + "…", key=ex):
        q = ex

if q:
    with st.spinner("Retrieving and answering…"):
        try:
            a = ask(q, method, k, rerank, mode)
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

    if a["abstained"]:
        st.warning("**Not enough evidence** in the retrieved abstracts to answer this. "
                   + a["answer"].replace(ABSTAIN, "").strip())
    else:
        st.markdown(linkify(a["answer"]))
        if a["invalid_citations"]:
            st.error(f"Cited PMIDs not in the retrieved context: {', '.join(a['invalid_citations'])}")
        if a["uncited_sentences"]:
            st.info(f"{a['uncited_sentences']} of {a['n_sentences']} sentences have no citation.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latency", f"{a['latency_s']:.2f}s")
    m2.metric("Tokens", a["input_tokens"] + a["output_tokens"])
    m3.metric("Cost", f"${a['cost_usd']:.5f}")
    m4.metric("Citations", len(a["citations"]))

    st.subheader("Retrieved abstracts")
    cited = set(a["citations"])
    for hit in a["hits"]:
        tag = "✅ cited · " if hit["pmid"] in cited else ""
        meta = " · ".join(str(x) for x in (hit.get("journal"), hit.get("year")) if x)
        cos = f" · cos {hit['dense_score']:.3f}" if hit.get("dense_score") is not None else ""
        with st.expander(f"{hit['rank']}. {tag}{hit['title'] or '(untitled)'}"):
            st.caption(f"[PMID {hit['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{hit['pmid']}/) · {meta} · "
                       f"BM25 rank {hit.get('bm25_rank') or '—'} · dense rank {hit.get('dense_rank') or '—'}{cos}")
            st.write(hit["abstract"])
