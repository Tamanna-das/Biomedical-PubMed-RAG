import os
import shutil

import pandas as pd
import pytest

from conftest import FIXTURES, ROOT


@pytest.fixture()
def eval_dir(demo_env, tmp_path, monkeypatch):
    d = tmp_path / "eval"
    d.mkdir()
    shutil.copy(ROOT / "eval" / "questions.jsonl", d / "questions.jsonl")
    shutil.copy(FIXTURES / "sample_gold.jsonl", d / "gold_pmids.jsonl")
    monkeypatch.setenv("BIOMED_EVAL_DIR", str(d))
    monkeypatch.setenv("BIOMED_RESULTS_DIR", str(tmp_path / "results"))
    return d


def test_eval_questions_file_is_valid():
    from src.io_utils import read_jsonl

    qs = read_jsonl(ROOT / "eval" / "questions.jsonl")
    assert len(qs) == 50 and len({q["qid"] for q in qs}) == 50
    assert sum(not q["answerable"] for q in qs) == 8


def test_retrieval_eval_writes_table(eval_dir, tmp_path):
    from src.eval import retrieval_eval

    retrieval_eval.main([])
    md = (tmp_path / "results" / "retrieval_own.md").read_text()
    assert "| bm25 |" in md and "| dense |" in md and "| hybrid |" in md
    df = pd.read_csv(tmp_path / "results" / "retrieval_own.csv")
    assert set(df["method"]) == {"bm25", "dense", "hybrid"} and df["recall@20"].min() > 0.5


def test_label_gold_pool_and_import(eval_dir):
    from src.eval import label_gold
    from src.io_utils import read_jsonl

    label_gold.main(["pool", "--depth", "3"])
    sheet = eval_dir / "pool_to_judge.csv"
    df = pd.read_csv(sheet, dtype=str).fillna("")
    assert set(df["qid"]) == {f"q{i:02d}" for i in range(1, 43)}  # unanswerable excluded
    df.loc[df.index[:2], "relevant"] = ["1", "0"]
    df.to_csv(sheet, index=False)
    label_gold.main(["pool", "--depth", "3"])  # re-pool keeps labels
    assert pd.read_csv(sheet, dtype=str).fillna("")["relevant"].tolist()[:2] == ["1", "0"]
    label_gold.main(["import"])
    gold = read_jsonl(eval_dir / "gold_pmids.jsonl")
    assert gold == [{"qid": df.loc[0, "qid"], "pmids": [df.loc[0, "pmid"]]}]


def test_answer_eval_run_and_summarize(eval_dir, tmp_path):
    from src.eval import answer_eval

    answer_eval.main(["run", "--sample", "5"])
    res = tmp_path / "results"
    md = (res / "answer_eval.md").read_text()
    assert "correct_abstention" in md and "grounded" in md and "plain" in md
    sheet = pd.read_csv(res / "faithfulness_sheet.csv", dtype=str).fillna("")
    sheet["label"] = ["supported"] * (len(sheet) - 1) + ["unsupported"]
    sheet.to_csv(res / "faithfulness_sheet.csv", index=False)
    answer_eval.main(["summarize"])
    assert "unsupported" in (res / "failures.md").read_text()


def test_api(demo_env):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from api.main import app
    from src.pipeline import get_pipeline

    get_pipeline.cache_clear()
    with TestClient(app) as c:
        h = c.get("/health").json()
        assert h["status"] == "ok" and h["documents"] == 16
        r = c.post("/ask", json={"question": "What do vanA genes do in vancomycin-resistant enterococci?"}).json()
        assert "99000008" in r["citations"] and not r["abstained"]
        assert c.post("/search", json={"query": "colistin mcr-1", "k": 2}).json()["hits"][0]["pmid"] == "99000001"
        assert c.post("/ask", json={"question": "x"}).status_code == 422


def test_pubmedqa_benchmark_build():
    from src.eval.pubmedqa import build_benchmark

    pqal = {"111": {"QUESTION": "Does X reduce Y?", "CONTEXTS": ["Part one.", "Part two."], "YEAR": "2014",
                    "final_decision": "yes"}}
    distr = pd.DataFrame([{"pmid": "222", "title": "T", "abstract": "A", "journal": "J", "year": 2020},
                          {"pmid": "111", "title": "dup", "abstract": "dup", "journal": "", "year": None}])
    df, qs, gold = build_benchmark(pqal, distr)
    assert df["pmid"].tolist() == ["111", "222"] and df.loc[0, "text"] == "Part one. Part two."
    assert qs[0]["qid"] == "pqa_111" and gold == [{"qid": "pqa_111", "pmids": ["111"]}]


def test_prelabel_fills_only_empty_rows():
    from src.eval.label_gold import prelabel
    from src.generate.llm import LLMResponse

    class Judge:
        def generate(self, msgs):
            return LLMResponse("1 - mentions mecA and PBP2a" if "Title: mecA" in msgs[1]["content"] else "0 - off topic", "j")

    df = pd.DataFrame({"qid": ["q1"] * 3, "question": ["What does mecA do?"] * 3, "pmid": ["1", "2", "3"],
                       "title": ["mecA in MRSA", "Efflux", "Biofilm"], "abstract": ["a", "b", "c"],
                       "relevant": ["", "", "1"], "llm_reason": ["", "", ""], "checked": ["", "", "y"]})
    out = prelabel(df, Judge())
    assert out["relevant"].tolist() == ["1", "0", "1"]
    assert out.loc[0, "llm_reason"] == "mentions mecA and PBP2a" and out.loc[2, "llm_reason"] == ""
