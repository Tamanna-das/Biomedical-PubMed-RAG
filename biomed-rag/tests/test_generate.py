from src.generate.answer import extract_citations, generate_answer, sentence_stats
from src.generate.llm import ExtractiveLLM, LLMResponse
from src.generate.prompt import ABSTAIN_TOKEN, build_messages


class FakeLLM:
    model = "fake"

    def __init__(self, text):
        self.text, self.calls = text, 0

    def generate(self, messages):
        self.calls += 1
        self.messages = messages
        return LLMResponse(self.text, "fake", 1000, 100)


def test_extract_citations_and_sentence_stats():
    t = "MRSA carries mecA [PMID:99000007]. It encodes PBP2a [PMID: 99000007, PMID:12345678]. This is an unsupported claim."
    assert extract_citations(t) == ["99000007", "12345678"]
    assert sentence_stats(t) == (3, 1)


def test_prompt_contains_numbered_context(retriever):
    hits = retriever.search("vancomycin resistance enterococci", "hybrid", 2)
    msgs = build_messages("Q?", hits, "grounded")
    assert msgs[0]["role"] == "system" and ABSTAIN_TOKEN in msgs[0]["content"]
    assert f"[PMID:{hits[0].pmid}]" in msgs[1]["content"] and msgs[1]["content"].endswith("citations):")


def test_invalid_citation_and_cost(retriever):
    hits = retriever.search("vancomycin resistance enterococci", "hybrid", 2)
    llm = FakeLLM(f"VRE carry vanA [PMID:{hits[0].pmid}]. They are common in hospitals [PMID:11111111].")
    a = generate_answer("q", hits, llm, price_in=1.0, price_out=2.0)
    assert not a.abstained and a.invalid_citations == ["11111111"]
    assert a.cost_usd == round(1000 / 1e6 * 1.0 + 100 / 1e6 * 2.0, 6)


def test_model_abstention_detected(retriever):
    hits = retriever.search("vancomycin", "bm25", 2)
    a = generate_answer("q", hits, FakeLLM(f"{ABSTAIN_TOKEN}\nThe abstracts do not discuss dosing."))
    assert a.abstained and a.abstain_reason == "model" and a.citations == []


def test_retrieval_gate_skips_llm(retriever):
    hits = retriever.search("vancomycin", "hybrid", 2)
    llm = FakeLLM("should not be called")
    a = generate_answer("q", hits, llm, abstain_threshold=1.01)
    assert a.abstained and a.abstain_reason == "retrieval_gate" and llm.calls == 0
    assert generate_answer("q", [], llm).abstained


def test_extractive_backend_answers_and_abstains(retriever):
    q = "How does the mecA gene make Staphylococcus aureus resistant to methicillin?"
    a = generate_answer(q, retriever.search(q, "hybrid", 3), ExtractiveLLM())
    assert not a.abstained and "99000007" in a.citations and not a.invalid_citations and a.uncited_sentences == 0
    q2 = "What dose of amoxicillin should I give my 3-year-old for an ear infection?"
    assert generate_answer(q2, retriever.search(q2, "hybrid", 3), ExtractiveLLM()).abstained
