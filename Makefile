PY ?= python
.PHONY: install install-dev test demo fetch clean-data index-bm25 index-dense index pool gold \
        eval-retrieval eval-pubmedqa eval-answers summarize api ui docker

install:        ; pip install -r requirements.txt
install-dev:    ; pip install -r requirements-dev.txt
test:           ; $(PY) -m pytest -q
demo:           ; $(PY) scripts/build_demo.py

# ---- Week 8: corpus + BM25
fetch:          ; $(PY) -m src.ingest.pubmed_fetch --max-docs 30000 --start-year 2010
clean-data:     ; $(PY) -m src.ingest.clean
index-bm25:     ; $(PY) -m src.index.bm25
# ---- Week 9: dense + hybrid
index-dense:    ; $(PY) -m src.index.dense
index: index-bm25 index-dense
pool:           ; $(PY) -m src.eval.label_gold pool
gold:           ; $(PY) -m src.eval.label_gold import
eval-retrieval: ; $(PY) -m src.eval.retrieval_eval
eval-pubmedqa:  ; $(PY) -m src.eval.pubmedqa --distractors 20000 && $(PY) -m src.eval.retrieval_eval --dataset pubmedqa
# ---- Week 10: generation
eval-answers:   ; $(PY) -m src.eval.answer_eval run
summarize:      ; $(PY) -m src.eval.answer_eval summarize
api:            ; uvicorn api.main:app --reload
ui:             ; streamlit run app/streamlit_app.py
docker:         ; docker compose up --build
