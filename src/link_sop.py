"""Step 3b-2: connect the mail side of the graph to the SOP side.

1. (Email)-[:CITES]->(Clause)          deterministic: "SOP-03 section 4.2" in a mail body
2. (Issue|Decision)-[:GOVERNED_BY]->(Clause)
      a) from sop_refs the extractor copied out of the mails (deterministic)
      b) otherwise: top-K clauses by vector similarity to the issue/decision summary,
         and Claude confirms which of the candidates actually govern the case.

Run:  uv run python src/link_sop.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import schema  # noqa: E402
from store import connect, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)
PRICE_IN, PRICE_OUT = 4.0, 20.0
CITE_RE = re.compile(r"\b(SOP-0\d)\s*(?:,|:)?\s*(?:section|sec\.|clause|§|para\.?|paragraph)?\s*(\d+\.\d+)\b", re.I)
TOP_K = 4


class Verdict(BaseModel):
    governing_clause_ids: list[str]
    reason: str


SYSTEM = """You are a compliance reviewer for a freight forwarder. You get one issue or decision from an e-mail thread and a few candidate SOP clauses. Return the ids of the clauses that actually govern this case (the rule the operator had to follow, or the rule the decision applies). Return an empty list if none of the candidates fits. Be strict: a clause governs the case only if its obligation is directly about this situation."""


def link_citations(conn) -> int:
    rows = run(conn, "MATCH (e:Email) RETURN e.id AS id, e.body AS body")
    pairs = []
    for r in rows:
        for doc, num in CITE_RE.findall(r["body"]):
            pairs.append({"email": r["id"], "clause": f"{doc.upper()}-C{num}"})
    res = run(conn, "UNWIND $rows AS r MATCH (e:Email {id: r.email}), (c:Clause {id: r.clause}) "
                    "MERGE (e)-[:CITES]->(c) RETURN count(*) AS n", {"rows": pairs})
    return res[0]["n"] if res else 0


def clause_label(c: dict) -> str:
    return f"{c['id']} ({c['doc']} {c['number']} {c['title']}): {c['text'][:400]}"


def main() -> None:
    conn = connect()
    n_cites = link_citations(conn)
    print(f"CITES links: {n_cites}")

    items = run(conn, "MATCH (n) WHERE n:Issue OR n:Decision RETURN n.id AS id, labels(n)[0] AS label, "
                      "n.summary AS summary, n.sop_refs AS refs, n.kind AS kind")
    embedder = SentenceTransformer(schema.EMBEDDING_MODEL)
    client = anthropic.Anthropic()
    tin = tout = 0
    explicit = inferred = 0
    log = []
    for it in items:
        # a) explicit refs copied from the mails
        ids = []
        for ref in it["refs"] or []:
            for doc, num in CITE_RE.findall(ref):
                ids.append(f"{doc.upper()}-C{num}")
        if ids:
            r = run(conn, "UNWIND $ids AS cid MATCH (n {id: $id}), (c:Clause {id: cid}) MERGE (n)-[:GOVERNED_BY]->(c) RETURN count(*) AS n",
                    {"id": it["id"], "ids": ids})
            explicit += r[0]["n"]
            log.append({"id": it["id"], "mode": "explicit", "clauses": ids})
            continue
        # b) vector candidates + LLM confirmation
        vec = embedder.encode(it["summary"], normalize_embeddings=True).tolist()
        cands = run(conn, f"MATCH (c:Clause) SEARCH c IN (VECTOR INDEX clause_embedding FOR vector($v, {schema.EMBEDDING_DIM}, FLOAT32) LIMIT $k) SCORE AS score "
                          "MATCH (d:SOPDocument)-[:HAS_SECTION]->(:Section)-[:HAS_CLAUSE]->(c) "
                          "RETURN c.id AS id, c.number AS number, c.title AS title, c.text AS text, d.id AS doc, score",
                    {"v": vec, "k": TOP_K})
        prompt = (f"{it['label']}: {it['summary']}\n\nCandidate clauses:\n" + "\n".join("- " + clause_label(c) for c in cands)
                  + "\n\nWhich candidate ids govern this case?")
        resp = client.messages.parse(model=schema.LLM_MODEL, max_tokens=600, system=SYSTEM,
                                     messages=[{"role": "user", "content": prompt}], output_format=Verdict)
        tin += resp.usage.input_tokens
        tout += resp.usage.output_tokens
        chosen = [c for c in resp.parsed_output.governing_clause_ids if c in {x["id"] for x in cands}]
        if chosen:
            r = run(conn, "UNWIND $ids AS cid MATCH (n {id: $id}), (c:Clause {id: cid}) MERGE (n)-[:GOVERNED_BY]->(c) RETURN count(*) AS n",
                    {"id": it["id"], "ids": chosen})
            inferred += r[0]["n"]
        log.append({"id": it["id"], "mode": "inferred", "candidates": [c["id"] for c in cands], "clauses": chosen,
                    "reason": resp.parsed_output.reason})
    usage = {"model": schema.LLM_MODEL, "items": len(items), "cites": n_cites, "governed_by_explicit": explicit,
             "governed_by_inferred": inferred, "input_tokens": tin, "output_tokens": tout,
             "cost_usd_list_price": round(tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT, 3)}
    (ROOT / "data" / "mail" / "_link_usage.json").write_text(json.dumps({"usage": usage, "log": log}, indent=1), encoding="utf-8")
    print(json.dumps(usage, indent=1))
    conn.close()


if __name__ == "__main__":
    main()
