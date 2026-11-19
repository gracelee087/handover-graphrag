"""Step 1b: write the five SOP documents (clause prose by LLM) and render them to PDF.

Structure (sections, clause numbers, titles, gists) comes from world/world.yaml.
The LLM only expands each gist into 2-5 sentences of policy text, and emits a
small table where the gist starts with "Table". Rendering uses fpdf2 with
real headings and tables so Docling has structure to recover in step 3b.

Outputs (data/sop/):
    json/SOP-0N.json       LLM output cache
    SOP-0N_<name>.md       Markdown source
    SOP-0N_<name>.pdf      the corpus document
    _usage.json

Run:  uv run python src/gen_sop.py
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import yaml
from dotenv import load_dotenv
from fpdf import FPDF
from fpdf.fonts import FontFace
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import schema  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)
W = yaml.safe_load((ROOT / "world" / "world.yaml").read_text(encoding="utf-8"))
OUT = ROOT / "data" / "sop"
PRICE_IN, PRICE_OUT = 4.0, 20.0


class Clause(BaseModel):
    clause_id: str
    text: str
    table_header: list[str] | None = None
    table_rows: list[list[str]] | None = None


class SopText(BaseModel):
    clauses: list[Clause]


SYSTEM = f"""You write Standard Operating Procedure text for {W['meta']['company']}, a sea-freight forwarder in Busan, Korea.
For each clause you receive a clause_id, a title and a gist. Expand the gist into formal SOP prose of 2-5 sentences
using "must" / "shall" for obligations, naming the responsible role (Import Operator, Import Team Lead, Operations Manager,
Documentation Clerk, Accounting). Keep every number, day count, USD amount, carrier/customer name and cross-reference
(e.g. "SOP-02") from the gist exactly; do not add new numbers or rules. Where the gist begins with "Table", also return
table_header and table_rows holding exactly the data in the gist, and keep the prose to one or two sentences that
introduce the table. Do not repeat the clause title or number in the text. Return one entry per clause_id, in order."""


def sop_prompt(sop: dict) -> str:
    lines = [f"Document: {sop['id']} {sop['title']} ({sop['version']}, effective {sop['effective_date']})", ""]
    for sec in sop["sections"]:
        lines.append(f"Section {sec['number']} {sec['title']}")
        for cl in sec["clauses"]:
            lines.append(f"  - clause_id {cl['id']} | {cl['number']} {cl['title']} | gist: {cl['gist']}")
    return "\n".join(lines)


def write_sop(client: anthropic.Anthropic, sop: dict) -> dict:
    cache = OUT / "json" / f"{sop['id']}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    resp = client.messages.parse(
        model=schema.LLM_MODEL, max_tokens=12000, system=SYSTEM,
        messages=[{"role": "user", "content": sop_prompt(sop)}], output_format=SopText,
    )
    got = {c.clause_id: c.model_dump() for c in resp.parsed_output.clauses}
    want = [cl["id"] for sec in sop["sections"] for cl in sec["clauses"]]
    missing = [c for c in want if c not in got]
    if missing:
        raise RuntimeError(f"{sop['id']}: missing clauses {missing}")
    rec = {"sop": sop["id"], "usage": {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens},
           "clauses": {c: got[c] for c in want}}
    cache.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec


def latin(s: str) -> str:
    """Core PDF fonts are Latin-1: fold typographic punctuation, drop the rest."""
    s = (s.replace("–", "-").replace("—", "-").replace("‘", "'").replace("’", "'")
         .replace("“", '"').replace("”", '"').replace("…", "...").replace(" ", " "))
    s = unicodedata.normalize("NFKD", s)
    return s.encode("latin-1", "ignore").decode("latin-1")


def to_markdown(sop: dict, rec: dict) -> str:
    md = [f"# {sop['title']}", "",
          f"**Document:** {sop['id']}  ", f"**Version:** {sop['version']}  ",
          f"**Effective date:** {sop['effective_date']}  ", f"**Owner:** Import Operations, {W['meta']['company']}", ""]
    for sec in sop["sections"]:
        md += [f"## {sec['number']} {sec['title']}", ""]
        for cl in sec["clauses"]:
            c = rec["clauses"][cl["id"]]
            md += [f"### {cl['number']} {cl['title']}", "", c["text"], ""]
            if c.get("table_header") and c.get("table_rows"):
                md.append("| " + " | ".join(c["table_header"]) + " |")
                md.append("|" + "---|" * len(c["table_header"]))
                for row in c["table_rows"]:
                    md.append("| " + " | ".join(row) + " |")
                md.append("")
    return "\n".join(md)


class SopPDF(FPDF):
    def __init__(self, sop: dict):
        super().__init__()
        self.sop = sop
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(20, 18, 20)

    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(110)
        self.cell(0, 6, latin(f"{W['meta']['company']}  |  {self.sop['id']} {self.sop['title']}  |  {self.sop['version']}"),
                  align="L", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_text_color(0)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(110)
        self.cell(0, 6, f"Page {self.page_no()}/{{nb}}  -  Controlled document. Printed copies are uncontrolled.", align="C")


def to_pdf(sop: dict, rec: dict, path: Path) -> None:
    pdf = SopPDF(sop)
    pdf.set_creation_date(datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc))   # deterministic bytes across re-runs
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 9, latin(sop["title"]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    for k, v in [("Document", sop["id"]), ("Version", sop["version"]), ("Effective date", sop["effective_date"]),
                 ("Owner", f"Import Operations, {W['meta']['company']}"), ("Approved by", "Operations Manager")]:
        pdf.cell(38, 6, f"{k}:", new_x="END")
        pdf.cell(0, 6, latin(v), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    for sec in sop["sections"]:
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, latin(f"{sec['number']}  {sec['title']}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        for cl in sec["clauses"]:
            c = rec["clauses"][cl["id"]]
            pdf.set_font("Helvetica", "B", 10.5)
            pdf.multi_cell(0, 6, latin(f"{cl['number']}  {cl['title']}"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5.5, latin(c["text"]), new_x="LMARGIN", new_y="NEXT")
            if c.get("table_header") and c.get("table_rows"):
                pdf.ln(1.5)
                pdf.set_font("Helvetica", "", 9)
                with pdf.table(borders_layout="ALL", line_height=5.5, text_align="LEFT",
                               headings_style=FontFace(emphasis="BOLD", fill_color=(230, 230, 230))) as table:
                    hdr = table.row()
                    for h in c["table_header"]:
                        hdr.cell(latin(h))
                    for row in c["table_rows"]:
                        r = table.row()
                        for cell in row:
                            r.cell(latin(cell))
            pdf.ln(3)
        pdf.ln(2)
    pdf.output(str(path))


def main() -> None:
    (OUT / "json").mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic()
    tin = tout = 0
    for sop in W["sops"]:
        rec = write_sop(client, sop)
        tin += rec["usage"]["input"]
        tout += rec["usage"]["output"]
        stem = re.sub(r"\.pdf$", "", sop["filename"])
        (OUT / f"{stem}.md").write_text(to_markdown(sop, rec), encoding="utf-8")
        to_pdf(sop, rec, OUT / sop["filename"])
        print(f"{sop['id']} ok -> {sop['filename']} ({len(rec['clauses'])} clauses)", flush=True)
    usage = {"model": schema.LLM_MODEL, "sops": len(W["sops"]), "input_tokens": tin, "output_tokens": tout,
             "cost_usd_list_price": round(tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT, 3)}
    (OUT / "_usage.json").write_text(json.dumps(usage, indent=1), encoding="utf-8")
    print(json.dumps(usage))


if __name__ == "__main__":
    main()
