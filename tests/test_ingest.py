from datetime import date

from conftest import FIXTURES
from src.ingest import pubmed_fetch as pf
from src.ingest.clean import clean_records
from src.io_utils import read_jsonl


def test_parse_pubmed_xml():
    recs = pf.parse_pubmed_xml((FIXTURES / "sample_efetch.xml").read_text())
    assert [r["pmid"] for r in recs] == ["99000101", "99000102"]
    a = recs[0]
    assert a["title"] == "Carbapenemase genes in Klebsiella pneumoniae isolates."  # inline <i> kept as text
    assert a["abstract"].startswith("BACKGROUND: Carbapenem resistance is rising.")
    assert "RESULTS: We detected blaKPC in 40% of isolates." in a["abstract"]
    assert a["year"] == 2021 and a["doi"] == "10.1000/test.1"
    assert a["mesh"] == ["Klebsiella pneumoniae"] and a["journal"] == "Journal of Test Microbiology"
    assert recs[1]["year"] == 2019  # from MedlineDate
    assert recs[1]["abstract"] == ""


def test_clean_drops_short_comments_and_duplicates():
    recs = pf.parse_pubmed_xml((FIXTURES / "sample_efetch.xml").read_text())
    df = clean_records(recs + recs[:1], min_chars=10)
    assert df["pmid"].tolist() == ["99000101"]
    assert df.loc[0, "text"].startswith("Carbapenemase genes in Klebsiella pneumoniae isolates. BACKGROUND")


class FakeClient:
    """Pretends one hit per day across the year; ESearch cap lowered to force bisection."""

    def __init__(self):
        self.calls = 0

    def count(self, term, mindate, maxdate):
        self.calls += 1
        d0, d1 = (date(*map(int, x.split("/"))) for x in (mindate, maxdate))
        return (d1 - d0).days + 1

    def search_ids(self, term, mindate, maxdate):
        d0, d1 = (date(*map(int, x.split("/"))) for x in (mindate, maxdate))
        return [str(d0.toordinal() + i) for i in range((d1 - d0).days + 1)]


def test_collect_ids_bisects_under_cap(monkeypatch):
    monkeypatch.setattr(pf, "ESEARCH_CAP", 40)
    ids = pf.collect_ids_for_range(FakeClient(), "q", date(2020, 1, 1), date(2020, 12, 31))
    assert len(ids) == 366 and len(set(ids)) == 366  # leap year, no gaps or duplicates


def test_collect_ids_respects_max_docs(monkeypatch):
    monkeypatch.setattr(pf, "ESEARCH_CAP", 1000)
    ids = pf.collect_ids(FakeClient(), "q", 2018, 2021, max_docs=500)
    assert len(ids) == 500


def test_fetch_records_resumes(tmp_path):
    xml = (FIXTURES / "sample_efetch.xml").read_text()

    class C:
        n = 0

        def fetch_xml(self, pmids):
            C.n += 1
            return xml

    out = tmp_path / "records.jsonl"
    pf.fetch_records(C(), ["99000101", "99000102"], out, batch_size=200)
    pf.fetch_records(C(), ["99000101", "99000102"], out, batch_size=200)  # nothing left to fetch
    assert C.n == 1 and len(read_jsonl(out)) == 2
