"""Download open-access PDFs for depth reads into data/local/pdfs (gitignored) and extract text to data/local/txt."""
import sys, time, urllib.request
from pathlib import Path
from pypdf import PdfReader
ROOT = Path(__file__).resolve().parents[2]
PDF, TXT = ROOT / "data/local/pdfs", ROOT / "data/local/txt"
SOURCES = {
 "moustapha2022active": "https://arxiv.org/pdf/2106.01713",
 "grinsztajn2022tree": "https://arxiv.org/pdf/2207.08815",
 "eggensperger2015efficient": "https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/15-AAAI-Surrogates.pdf",
 "fuhg2021state": "https://hal.science/hal-02919440/document",
 "luthen2021sparse": "https://arxiv.org/pdf/2002.01290",
 "mainini2022analytical": "https://arxiv.org/pdf/2204.07867",
 "peherstorfer2018survey": "https://cims.nyu.edu/~pehersto/preprints/multi-fidelity-survey-peherstorfer-willcox-gunzburger.pdf",
 "ober2021promises": "https://arxiv.org/pdf/2102.12108",
 "lutjens2024climatebench": "https://arxiv.org/pdf/2408.05288",
 "forrester2007multi": "https://eprints.soton.ac.uk/64698/1/RSPA20071900.pdf",
 "vanrijn2021tradeoffs": "https://arxiv.org/pdf/2103.03280",
 "pse2024sbo": "https://arxiv.org/pdf/2412.13948",
 "legratiet2014recursive": "https://arxiv.org/pdf/1210.0686",
 "brevault2020overview": "https://arxiv.org/pdf/2006.16728",
 "eggensperger2018efficient": "https://arxiv.org/pdf/1703.10342",
 "lamperti2018agent": "https://arxiv.org/pdf/1703.10639",
 "hollmann2025tabpfn": "https://www.nature.com/articles/s41586-024-08328-6.pdf",
 "wilson2016deep": "https://arxiv.org/pdf/1511.02222",
 "hokanson2018data": "https://arxiv.org/pdf/1702.05859",
 "hampton2015compressive": "https://arxiv.org/pdf/1408.4157",
}
def main(keys):
    for k in keys or SOURCES:
        p = PDF / f"{k}.pdf"
        if not p.exists():
            try:
                req = urllib.request.Request(SOURCES[k], headers={"User-Agent": "Mozilla/5.0 (research)"})
                p.write_bytes(urllib.request.urlopen(req, timeout=60).read()); time.sleep(2)
            except Exception as e:
                print("FAIL download", k, e); continue
    for p in sorted(PDF.glob("*.pdf")):
        t = TXT / f"{p.stem}.txt"
        if t.exists(): continue
        try:
            r = PdfReader(p); t.write_text("\n".join((pg.extract_text() or "") for pg in r.pages), encoding="utf8")
            print(f"{p.stem:30s} {len(r.pages):3d} pages")
        except Exception as e:
            print("FAIL parse", p.stem, e)
if __name__ == "__main__":
    main(sys.argv[1:])
