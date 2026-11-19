"""Rebuild the whole knowledge graph from the committed corpus into data/handover.rl.

    uv run python build_all.py            # reuse LLM caches (data/mail/extract, data/sop/json)
    uv run python build_all.py --fresh    # delete the store first (same result, from zero)

Steps (each is a standalone script in src/):
    store.py          constraints + vector/full-text indexes
    load_structure.py directory, shipments, parsed .eml -> Person/Email/Thread/Shipment
    parse_sop.py      Docling on the SOP PDFs -> SOPDocument/Section/Clause (+ embeddings)
    extract_mail.py   Claude: Issue/Decision/Task/Deadline per thread (needs ANTHROPIC_API_KEY unless cached)
    dedupe_tasks.py   same owner+due+shipment in a later thread -> status 'duplicate', DUPLICATE_OF
    link_sop.py       Email-CITES->Clause, Issue/Decision-GOVERNED_BY->Clause
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["store.py", "load_structure.py", "parse_sop.py", "extract_mail.py", "dedupe_tasks.py", "link_sop.py"]

if "--fresh" in sys.argv:
    for f in (ROOT / "data").glob("handover.rl*"):
        f.unlink()
        print("removed", f.name)
for step in STEPS:
    print(f"\n==== {step} ====", flush=True)
    r = subprocess.run([sys.executable, str(ROOT / "src" / step)], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"{step} failed with exit code {r.returncode}")
print("\nbuild complete:", ROOT / "data" / "handover.rl")
