"""Check that every \\cite key in paper/sections/*.tex exists in paper/refs.bib."""
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[2] / "paper"
bib = set(re.findall(r"@\w+\{([^,]+),", (PAPER / "refs.bib").read_text(encoding="utf8")))
used = set()
for f in sorted((PAPER / "sections").glob("*.tex")):
    for group in re.findall(r"\\cite[pt]?(?:\[[^\]]*\])*\{([^}]+)\}", f.read_text(encoding="utf8")):
        used |= {k.strip() for k in group.split(",")}
print(f"cited {len(used)} / bib {len(bib)}")
print("missing from bib:", sorted(used - bib) or "none")
print("in bib but not yet cited:", len(bib - used))
