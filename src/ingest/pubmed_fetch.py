"""Fetch PubMed abstracts through NCBI E-utilities.

Why the date slicing: ESearch can only page through the first 9,999 PMIDs of any query,
so a broad topic query is split into date ranges small enough to stay under that cap
(ranges are bisected automatically until each one fits).

Usage:
    python -m src.ingest.pubmed_fetch --max-docs 30000 --start-year 2010 --end-year 2025
Resumable: PMIDs are cached in data/raw/pmids.txt and fetched records are appended to
data/raw/records.jsonl, so re-running continues where it stopped.
"""
from __future__ import annotations

import argparse
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path

import requests

from src.config import get_settings
from src.io_utils import read_jsonl, setup_logging, write_jsonl

log = logging.getLogger("pubmed_fetch")

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_CAP = 9999

DEFAULT_QUERY = (
    '("Drug Resistance, Bacterial"[MeSH] OR "Drug Resistance, Microbial"[MeSH] '
    'OR "antimicrobial resistance"[tiab] OR "antibiotic resistance"[tiab] '
    'OR "multidrug resistant"[tiab] OR carbapenemase[tiab] OR "extended-spectrum beta-lactamase"[tiab]) '
    "AND hasabstract AND english[lang]"
)


class EUtilsClient:
    def __init__(self, email: str = "", api_key: str = "", session: requests.Session | None = None,
                 max_retries: int = 5):
        self.session = session or requests.Session()
        self.email = email
        self.api_key = api_key
        self.delay = 0.11 if api_key else 0.34  # 10 req/s with key, 3 req/s without
        self.max_retries = max_retries
        self._last = 0.0

    def _params(self, extra: dict) -> dict:
        p = {"db": "pubmed", "tool": "biomed-rag", **extra}
        if self.email:
            p["email"] = self.email
        if self.api_key:
            p["api_key"] = self.api_key
        return p

    def _request(self, endpoint: str, params: dict, post: bool = False) -> requests.Response:
        url = f"{EUTILS}/{endpoint}"
        for attempt in range(self.max_retries):
            wait = self.delay - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                if post:
                    r = self.session.post(url, data=self._params(params), timeout=60)
                else:
                    r = self.session.get(url, params=self._params(params), timeout=60)
                if r.status_code == 200:
                    return r
                log.warning("%s HTTP %s (attempt %d)", endpoint, r.status_code, attempt + 1)
            except requests.RequestException as e:
                log.warning("%s failed: %s (attempt %d)", endpoint, e, attempt + 1)
            time.sleep(2 ** attempt)
        raise RuntimeError(f"E-utilities {endpoint} failed after {self.max_retries} attempts")

    def count(self, term: str, mindate: str, maxdate: str) -> int:
        r = self._request("esearch.fcgi", {"term": term, "datetype": "pdat", "mindate": mindate,
                                           "maxdate": maxdate, "retmax": 0, "retmode": "json"})
        return int(r.json()["esearchresult"]["count"])

    def search_ids(self, term: str, mindate: str, maxdate: str) -> list[str]:
        r = self._request("esearch.fcgi", {"term": term, "datetype": "pdat", "mindate": mindate,
                                           "maxdate": maxdate, "retmax": ESEARCH_CAP, "retmode": "json"})
        return r.json()["esearchresult"]["idlist"]

    def fetch_xml(self, pmids: list[str]) -> str:
        r = self._request("efetch.fcgi", {"id": ",".join(pmids), "retmode": "xml"}, post=True)
        return r.text


def _fmt(d: date) -> str:
    return d.strftime("%Y/%m/%d")


def collect_ids_for_range(client: EUtilsClient, term: str, start: date, end: date) -> list[str]:
    """Return all PMIDs in [start, end], bisecting the range while it exceeds the ESearch cap."""
    n = client.count(term, _fmt(start), _fmt(end))
    if n == 0:
        return []
    if n <= ESEARCH_CAP or start == end:
        if n > ESEARCH_CAP:
            log.warning("%s has %d hits on a single day; keeping the first %d", start, n, ESEARCH_CAP)
        ids = client.search_ids(term, _fmt(start), _fmt(end))
        log.info("%s..%s: %d ids", start, end, len(ids))
        return ids
    mid = start + (end - start) // 2
    return (collect_ids_for_range(client, term, start, mid)
            + collect_ids_for_range(client, term, mid + timedelta(days=1), end))


def collect_ids(client: EUtilsClient, term: str, start_year: int, end_year: int, max_docs: int) -> list[str]:
    """Newest years first, until max_docs unique PMIDs are collected."""
    seen: dict[str, None] = {}
    for year in range(end_year, start_year - 1, -1):
        for pmid in collect_ids_for_range(client, term, date(year, 1, 1), date(year, 12, 31)):
            seen.setdefault(pmid, None)
        log.info("after %d: %d unique PMIDs", year, len(seen))
        if len(seen) >= max_docs:
            break
    return list(seen)[:max_docs]


# ---------------------------------------------------------------- XML parsing

def _text(el: ET.Element | None) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip() if el is not None else ""


def _year(article: ET.Element) -> int | None:
    for path in ("MedlineCitation/Article/Journal/JournalIssue/PubDate/Year",
                 "MedlineCitation/Article/ArticleDate/Year"):
        el = article.find(path)
        if el is not None and el.text and el.text.strip().isdigit():
            return int(el.text.strip())
    md = article.find("MedlineCitation/Article/Journal/JournalIssue/PubDate/MedlineDate")
    if md is not None and md.text:
        m = re.search(r"(19|20)\d{2}", md.text)
        if m:
            return int(m.group(0))
    return None


def parse_pubmed_xml(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    out = []
    for art in root.iter("PubmedArticle"):
        mc = art.find("MedlineCitation")
        if mc is None:
            continue
        pmid = _text(mc.find("PMID"))
        a = mc.find("Article")
        if a is None or not pmid:
            continue
        parts = []
        for at in a.findall("Abstract/AbstractText"):
            t = _text(at)
            if not t:
                continue
            label = at.get("Label")
            parts.append(f"{label.strip().upper()}: {t}" if label else t)
        doi = ""
        for aid in art.findall("PubmedData/ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi":
                doi = _text(aid)
        out.append({
            "pmid": pmid,
            "title": _text(a.find("ArticleTitle")),
            "abstract": " ".join(parts),
            "journal": _text(a.find("Journal/Title")),
            "year": _year(art),
            "doi": doi,
            "mesh": [_text(d) for d in mc.findall("MeshHeadingList/MeshHeading/DescriptorName")],
            "pub_types": [_text(p) for p in a.findall("PublicationTypeList/PublicationType")],
        })
    return out


# ---------------------------------------------------------------- driver

def fetch_records(client: EUtilsClient, pmids: list[str], out_path: Path, batch_size: int = 200) -> int:
    done = {r["pmid"] for r in read_jsonl(out_path)}
    todo = [p for p in pmids if p not in done]
    log.info("%d PMIDs total, %d already fetched, %d to fetch", len(pmids), len(done), len(todo))
    n = 0
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        records = parse_pubmed_xml(client.fetch_xml(batch))
        write_jsonl(records, out_path, append=True)
        n += len(records)
        if (i // batch_size) % 10 == 0:
            log.info("fetched %d / %d", n, len(todo))
    return n


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--max-docs", type=int, default=30000)
    ap.add_argument("--start-year", type=int, default=2010)
    ap.add_argument("--end-year", type=int, default=date.today().year - 1)
    ap.add_argument("--out", type=Path, default=s.raw_path)
    ap.add_argument("--refresh-ids", action="store_true", help="ignore cached pmids.txt")
    args = ap.parse_args(argv)

    if not s.ncbi_email:
        log.warning("Set NCBI_EMAIL in .env; NCBI asks for a contact address.")
    client = EUtilsClient(email=s.ncbi_email, api_key=s.ncbi_api_key)

    ids_path = args.out.parent / "pmids.txt"
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    if ids_path.exists() and not args.refresh_ids:
        pmids = ids_path.read_text().split()
        log.info("loaded %d cached PMIDs from %s", len(pmids), ids_path)
    else:
        pmids = collect_ids(client, args.query, args.start_year, args.end_year, args.max_docs)
        ids_path.write_text("\n".join(pmids))
        (args.out.parent / "query.txt").write_text(
            f"{args.query}\nyears={args.start_year}-{args.end_year} max_docs={args.max_docs}\n")
    fetch_records(client, pmids, args.out)
    log.info("done -> %s", args.out)


if __name__ == "__main__":
    main()
