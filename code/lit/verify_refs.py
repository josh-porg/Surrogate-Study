"""Verify the literature seed list against Crossref (and arXiv) and build refs.bib.

Input : data/literature/refs_seed.csv   (key|query|arxiv|section)
Output: data/literature/verified.json    one record per key with the best match + a match score
        paper/refs.bib                    BibTeX from DOI content negotiation (or arXiv @misc)

A match is accepted when the share of the query's title words found in the candidate title is high
(score >= 0.6). Everything else is flagged for manual review — nothing is silently accepted.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "data" / "literature" / "refs_seed.csv"
OUT_JSON = ROOT / "data" / "literature" / "verified.json"
OUT_BIB = ROOT / "paper" / "refs.bib"
UA = {"User-Agent": "SurrogatePaper-litcheck/0.1 (research bibliography verification)"}
STOP = set("a an the of and for in on to with via by from at as is are its using use their".split())


def _get(url: str, accept: str | None = None, timeout: float = 30.0) -> bytes:
    hdr = dict(UA)
    if accept:
        hdr["Accept"] = accept
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=timeout) as r:
                return r.read()
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed: {url}")


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 2}


def _score(query: str, title: str) -> float:
    """Fraction of the candidate title's content words that appear in the query (author names in
    the query do not hurt because we normalise by the title)."""
    t = _words(title)
    return len(t & _words(query)) / max(len(t), 1)


def crossref(query: str) -> dict | None:
    url = "https://api.crossref.org/works?rows=5&query.bibliographic=" + urllib.parse.quote(query)
    items = json.loads(_get(url))["message"]["items"]
    best = None
    for it in items:
        title = (it.get("title") or [""])[0]
        sc = _score(query, title)
        if best is None or sc > best["score"]:
            yr = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
            best = dict(source="crossref", doi=it.get("DOI"), title=title, year=yr,
                        venue=(it.get("container-title") or [""])[0] or it.get("publisher", ""),
                        authors=[a.get("family", "") for a in it.get("author", [])][:6], score=round(sc, 2),
                        type=it.get("type"))
    return best


_ARXIV_LOCK = __import__("threading").Lock()


def arxiv(aid: str) -> dict | None:
    with _ARXIV_LOCK:                  # arXiv API etiquette: one request every ~3 s
        xml = _get("https://export.arxiv.org/api/query?id_list=" + aid)
        time.sleep(3.0)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    e = ET.fromstring(xml).find("a:entry", ns)
    if e is None or e.find("a:title", ns) is None:
        return None
    title = " ".join(e.find("a:title", ns).text.split())
    authors = [a.find("a:name", ns).text for a in e.findall("a:author", ns)]
    yr = int(e.find("a:published", ns).text[:4])
    doi = e.find("{http://arxiv.org/schemas/atom}doi")
    return dict(source="arxiv", arxiv=aid, title=title, year=yr, authors=[a.split()[-1] for a in authors][:6],
                authors_full=authors, doi=doi.text if doi is not None else None)


def crossref_doi(doi: str) -> dict:
    it = json.loads(_get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
    yr = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
    return dict(source="crossref", doi=it.get("DOI"), title=(it.get("title") or [""])[0], year=yr,
                venue=(it.get("container-title") or [""])[0] or it.get("publisher", ""),
                authors=[a.get("family", "") for a in it.get("author", [])][:6], score=1.0, type=it.get("type"),
                forced=True)


def check(row: dict) -> dict:
    rec = dict(key=row["key"], query=row["query"], section=row["section"])
    if row.get("doi"):                 # DOI pinned by hand after the audit: trust it, record its metadata
        try:
            rec["best"] = rec["crossref"] = crossref_doi(row["doi"])
            rec["status"] = "verified"
            return rec
        except Exception as ex:
            rec["error"] = str(ex)
    try:
        cr = crossref(row["query"])
    except Exception as ex:          # network hiccup: record, do not crash the batch
        cr = None
        rec["error"] = str(ex)
    ax = None
    if row.get("arxiv"):
        try:
            ax = arxiv(row["arxiv"])
            if ax:
                ax["score"] = round(_score(row["query"], ax["title"]), 2)
        except Exception as ex:
            rec["arxiv_error"] = str(ex)
    rec["crossref"], rec["arxiv"] = cr, ax
    cands = [c for c in (cr, ax) if c]
    best = max(cands, key=lambda c: c["score"]) if cands else None
    # prefer the published DOI only when it matches at least as well as the arXiv record (audit 2026-10-03:
    # the old "within 0.15" rule picked wrong Crossref hits, e.g. KAN, FNO, DKL)
    if row.get("prefer") == "arxiv" and ax:
        best = ax
    elif cr and ax and cr["score"] >= 0.6 and cr["score"] >= ax["score"]:
        best = cr
    rec["best"] = best
    rec["status"] = "verified" if best and best["score"] >= 0.6 else "review"
    return rec


def bibtex(rec: dict) -> str:
    b = rec["best"]
    if b and b.get("doi") and b["source"] == "crossref":
        try:
            txt = _get("https://doi.org/" + urllib.parse.quote(b["doi"]), accept="application/x-bibtex").decode("utf8")
            return re.sub(r"^@(\w+)\{[^,]*,", lambda m: "@" + m.group(1) + "{" + rec["key"] + ",", txt.strip(), count=1)
        except Exception:
            pass
    if b and b["source"] == "arxiv":
        au = " and ".join(b["authors_full"])
        return (f"@misc{{{rec['key']},\n  title = {{{b['title']}}},\n  author = {{{au}}},\n  year = {{{b['year']}}},\n"
                f"  eprint = {{{b['arxiv']}}},\n  archivePrefix = {{arXiv}}\n}}")
    return f"% UNVERIFIED {rec['key']}: {rec['query']}"


MANUAL = ROOT / "data" / "literature" / "manual_refs.bib"   # hand-checked entries not indexed online

_MONTHS = {m: m[:3] for m in ("january february march april may june july august september "
                              "october november december").split()} | {"sept": "sep"}
_UNICODE = {"′": "'", "’": "'", "‘": "`", "–": "--", "—": "---", "ℓ": r"$\ell$",
            "‐": "-", " ": " "}


def sanitize(bib: str) -> str:
    """Make Crossref BibTeX compile under plain BibTeX/pdflatex: month macros, HTML tags from JATS titles
    (<i>, <sub>…), stray Unicode punctuation, and collapsed whitespace inside fields."""
    bib = re.sub(r"month\s*=\s*\{?(\w+)\}?", lambda m: "month=" + _MONTHS.get(m.group(1).lower(), m.group(1)), bib)
    bib = re.sub(r"</?(i|b|sub|sup|scp|mml:[a-z]+|inline-formula)[^>]*>", "", bib)
    for u, a in _UNICODE.items():
        bib = bib.replace(u, a)
    return re.sub(r"[ \t]*\n[ \t]+(?=[^@\s])", " ", bib)


def _manual_keys() -> set[str]:
    return set(re.findall(r"@\w+\{([^,]+),", MANUAL.read_text(encoding="utf8"))) if MANUAL.exists() else set()


def main(argv):
    """`python verify_refs.py`            re-check only keys not yet verified (incremental)
       `python verify_refs.py --all`      re-check everything
       `python verify_refs.py key1 key2`  re-check the named keys"""
    rows = list(csv.DictReader(SEED.open(encoding="utf8"), delimiter="|"))
    old = {r["key"]: r for r in json.loads(OUT_JSON.read_text(encoding="utf8"))} if OUT_JSON.exists() else {}
    manual = _manual_keys()
    if "--all" in argv:
        todo = rows
    elif len(argv) > 1:
        todo = [r for r in rows if r["key"] in argv[1:]]
    else:
        todo = [r for r in rows if r["key"] not in manual
                and old.get(r["key"], {}).get("status") != "verified"]
    with ThreadPoolExecutor(6) as ex:
        for rec in ex.map(check, todo):
            old[rec["key"]] = rec
    for k in manual:                     # manual entries count as verified (checked by hand)
        old.setdefault(k, dict(key=k))["status"] = "manual"
    recs = [old[r["key"]] for r in rows if r["key"] in old]
    with ThreadPoolExecutor(6) as ex:
        bibs = list(ex.map(lambda r: "" if r["status"] == "manual" else bibtex(r), recs))
    OUT_JSON.write_text(json.dumps(recs, indent=1, ensure_ascii=False), encoding="utf8")
    OUT_BIB.parent.mkdir(parents=True, exist_ok=True)
    extra = MANUAL.read_text(encoding="utf8") if MANUAL.exists() else ""
    OUT_BIB.write_text(sanitize("\n\n".join(b for b in bibs if b)) + "\n\n" + extra, encoding="utf8")
    nv = sum(r["status"] in ("verified", "manual") for r in recs)
    print(f"{nv}/{len(recs)} verified; review list:")
    for r in recs:
        if r["status"] not in ("verified", "manual"):
            b = r.get("best") or {}
            print(f"  {r['key']:32s} score={b.get('score')}  -> {b.get('title','')[:80]!r} ({b.get('year')})")


if __name__ == "__main__":
    main(sys.argv)
