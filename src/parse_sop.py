"""Step 3b-1: SOP PDFs -> Docling -> SOPDocument / Section / Clause nodes (+ clause embeddings).

Docling recovers headings and tables from the PDF layout; clause boundaries are
then detected from the numbering ("4.2 Extension timing"). The clause text keeps
any table as Markdown so the retrievers can quote it.

Verification against world.yaml (the generator knows how many clauses exist)
is printed at the end: recall must be >= 95 % (PLAN step 3b).

Run:  uv run python src/parse_sop.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml
from docling.document_converter import DocumentConverter
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import schema  # noqa: E402
from store import connect, init_schema, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
W = yaml.safe_load((ROOT / "world" / "world.yaml").read_text(encoding="utf-8"))
SOP_DIR = ROOT / "data" / "sop"
DOCLING_OUT = SOP_DIR / "docling"

SEC_RE = re.compile(r"^#{1,6}\s*(\d+)\s+(.+?)\s*$")
CL_RE = re.compile(r"^#{1,6}\s*(\d+\.\d+)\s+(.+?)\s*$")
META_RE = re.compile(r"^(Document|Version|Effective date|Owner|Approved by):\s*(.+)$")


def pdf_to_markdown(pdf: Path, converter: DocumentConverter) -> str:
    DOCLING_OUT.mkdir(parents=True, exist_ok=True)
    cached = DOCLING_OUT / (pdf.stem + ".md")
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    md = converter.convert(str(pdf)).document.export_to_markdown()
    cached.write_text(md, encoding="utf-8")
    return md


def parse_markdown(md: str, doc_id: str) -> dict:
    """-> {title, version, effective_date, sections: [{number,title,clauses:[{number,title,text}]}]}"""
    doc = {"title": None, "version": None, "effective_date": None, "sections": []}
    sec = cl = None
    for raw in md.splitlines():
        line = raw.strip()
        if not line:
            continue
        if m := META_RE.match(line):
            key, val = m.group(1), m.group(2).strip()
            if key == "Version":
                doc["version"] = val
            elif key == "Effective date":
                doc["effective_date"] = val
            continue
        if (m := CL_RE.match(line)) and sec is not None:
            cl = {"number": m.group(1), "title": m.group(2), "text": []}
            sec["clauses"].append(cl)
            continue
        if m := SEC_RE.match(line):
            sec = {"number": m.group(1), "title": m.group(2), "clauses": []}
            doc["sections"].append(sec)
            cl = None
            continue
        if line.startswith("#"):
            if doc["title"] is None and not sec:
                doc["title"] = line.lstrip("#").strip()
            continue
        if doc["title"] is None and not sec:
            doc["title"] = line
            continue
        # running header/footer lines Docling may keep
        if line.startswith("Page ") or line.startswith(W["meta"]["company"] + "  |"):
            continue
        if cl is not None:
            cl["text"].append(line)
    for s in doc["sections"]:
        for c in s["clauses"]:
            c["text"] = "\n".join(c["text"]).strip()
    return doc


def load(conn, doc_id: str, filename: str, doc: dict, embedder: SentenceTransformer) -> int:
    run(conn, "MERGE (d:SOPDocument {id: $id}) SET d.title = $title, d.version = $version, "
              "d.effective_date = $eff, d.filename = $fn",
        {"id": doc_id, "title": doc["title"], "version": doc["version"], "eff": doc["effective_date"], "fn": filename})
    sec_rows, cl_rows = [], []
    for s in doc["sections"]:
        sid = f"{doc_id}-S{s['number']}"
        sec_rows.append({"id": sid, "number": s["number"], "title": s["title"]})
        for c in s["clauses"]:
            cl_rows.append({"id": f"{doc_id}-C{c['number']}", "sec": sid, "number": c["number"],
                            "title": c["title"], "text": c["text"]})
    run(conn, "UNWIND $rows AS r MATCH (d:SOPDocument {id: $doc}) MERGE (s:Section {id: r.id}) "
              "SET s.number = r.number, s.title = r.title MERGE (d)-[:HAS_SECTION]->(s)",
        {"rows": sec_rows, "doc": doc_id})
    texts = [f"{doc['title']} - {r['number']} {r['title']}. {r['text']}" for r in cl_rows]
    vecs = embedder.encode(texts, normalize_embeddings=True).tolist()
    for r, v in zip(cl_rows, vecs):
        r["emb"] = v
    run(conn, "UNWIND $rows AS r MATCH (s:Section {id: r.sec}) MERGE (c:Clause {id: r.id}) "
              "SET c.number = r.number, c.title = r.title, c.text = r.text, "
              f"c.embedding = vector(r.emb, {schema.EMBEDDING_DIM}, FLOAT32) "
              "MERGE (s)-[:HAS_CLAUSE]->(c)", {"rows": cl_rows})
    for label, rel, rows in [("Section", "NEXT_SECTION", sec_rows), ("Clause", "NEXT_CLAUSE", cl_rows)]:
        pairs = [{"a": rows[i]["id"], "b": rows[i + 1]["id"]} for i in range(len(rows) - 1)]
        run(conn, f"UNWIND $rows AS r MATCH (a:{label} {{id: r.a}}), (b:{label} {{id: r.b}}) MERGE (a)-[:{rel}]->(b)",
            {"rows": pairs})
    return len(cl_rows)


def main() -> None:
    converter = DocumentConverter()
    embedder = SentenceTransformer(schema.EMBEDDING_MODEL)
    conn = connect()
    init_schema(conn)
    expected = {c["id"]: c for sop in W["sops"] for sec in sop["sections"] for c in sec["clauses"]}
    found: set[str] = set()
    report = {}
    for sop in W["sops"]:
        pdf = SOP_DIR / sop["filename"]
        md = pdf_to_markdown(pdf, converter)
        doc = parse_markdown(md, sop["id"])
        n = load(conn, sop["id"], sop["filename"], doc, embedder)
        ids = {f"{sop['id']}-C{c['number']}" for s in doc["sections"] for c in s["clauses"]}
        found |= ids
        report[sop["id"]] = {"sections": len(doc["sections"]), "clauses": n,
                             "empty_clauses": [c["number"] for s in doc["sections"] for c in s["clauses"] if not c["text"]]}
        print(sop["id"], report[sop["id"]], flush=True)
    missing = sorted(set(expected) - found)
    extra = sorted(found - set(expected))
    recall = len(found & set(expected)) / len(expected)
    print(f"clauses expected={len(expected)} found={len(found)} recall={recall:.1%} missing={missing} extra={extra}")
    (SOP_DIR / "_parse_report.json").write_text(
        json.dumps({"per_sop": report, "recall": recall, "missing": missing, "extra": extra}, indent=1), encoding="utf-8")
    conn.close()
    if recall < 0.95:
        sys.exit(1)


if __name__ == "__main__":
    main()
