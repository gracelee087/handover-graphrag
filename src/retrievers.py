"""Step 4: three retrieval strategies over the Relate store, one answerer.

  vector        question -> top-k mail chunks by vector similarity -> answer
  graph         same top-k chunks, then expand each hit along the graph:
                the mail's thread/sender/recipients/shipment, the issues,
                decisions and tasks extracted from it (with owners), and the
                SOP clauses they are governed by or cite            -> answer
  text2cypher   Claude writes one Cypher query against the schema, Relate
                runs it (read-only), the rows become the context    -> answer

All three use the same embedding model (schema.EMBEDDING_MODEL) and the same
answer prompt, so the comparison in eval/run_eval.py isolates the retrieval step.

Usage:  uv run python src/retrievers.py "Who decided the demurrage split on HBL MFCIM260009?" [--mode graph]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import schema  # noqa: E402
from store import connect, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)
TOP_K = 6
MODES = ("vector", "graph", "text2cypher")


@dataclass
class Result:
    mode: str
    question: str
    answer: str
    context: str
    cypher: str | None = None
    retrieved: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    error: str | None = None


# --------------------------------------------------------------------------- shared
_embedder: SentenceTransformer | None = None


def embed(text: str) -> list[float]:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(schema.EMBEDDING_MODEL)
    return _embedder.encode(text, normalize_embeddings=True).tolist()


def vector_hits(conn, question: str, k: int = TOP_K) -> list[dict]:
    return run(conn, f"""
        MATCH (c:{schema.CHUNK_LABEL})
        SEARCH c IN (VECTOR INDEX chunk_embedding FOR vector($v, {schema.EMBEDDING_DIM}, FLOAT32) LIMIT $k) SCORE AS score
        MATCH (c)-[:{schema.REL_FROM_DOCUMENT}]->(e:Email)<-[:SENT]-(p:Person)
        RETURN c.id AS chunk, e.id AS email, e.date AS date, p.name AS sender, e.subject AS subject,
               c.{schema.CHUNK_TEXT_PROP} AS text, score
        ORDER BY score DESC""", {"v": embed(question), "k": k})


ANSWER_SYSTEM = """You answer questions for a new import operator who inherited a predecessor's mailbox and the company SOPs.
Answer ONLY from the context given. Be concrete: names, dates (YYYY-MM-DD), amounts, HBL/MBL numbers, SOP clause numbers.
If the context does not contain the answer, say "Not found in the retrieved context" and state what is missing.
Cite evidence in brackets, e.g. [M-0231] for a mail or [SOP-03 4.2] for a clause. Keep the answer under 120 words."""


def answer(client: anthropic.Anthropic, question: str, context: str) -> tuple[str, dict]:
    resp = client.messages.create(
        model=schema.LLM_MODEL, max_tokens=800, system=ANSWER_SYSTEM,
        messages=[{"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text, {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}


# --------------------------------------------------------------------------- 1. vector
def retrieve_vector(conn, question: str) -> tuple[str, list[str], None]:
    hits = vector_hits(conn, question)
    ctx = "\n\n".join(f"[{h['email']}] {h['date'][:10]} from {h['sender']} (score {h['score']:.2f})\n{h['text']}" for h in hits)
    return ctx, [h["email"] for h in hits], None


# --------------------------------------------------------------------------- 2. graph-expanded
EXPAND = f"""
UNWIND $emails AS eid
MATCH (e:Email {{id: eid}})<-[:SENT]-(p:Person)-[:WORKS_AT]->(po:Organization)
OPTIONAL MATCH (e)-[:TO]->(t:Person)
OPTIONAL MATCH (e)-[:ABOUT]->(s:Shipment)
OPTIONAL MATCH (s)-[:CONSIGNEE]->(cons:Organization)
OPTIONAL MATCH (s)-[:CARRIED_BY]->(car:Organization)
OPTIONAL MATCH (s)-[:CLEARED_BY]->(brk:Organization)
OPTIONAL MATCH (e)-[:IN_THREAD]->(th:Thread)
OPTIONAL MATCH (e)-[:CITES]->(cc:Clause)<-[:HAS_CLAUSE]-(:Section)<-[:HAS_SECTION]-(cd:SOPDocument)
WITH e, p, po, collect(DISTINCT t.name) AS recipients, s, cons, car, brk, th,
     collect(DISTINCT {{id: cc.id, doc: cd.id, number: cc.number, title: cc.title, text: cc.text}}) AS cites
OPTIONAL MATCH (e)<-[:{schema.REL_FROM_DOCUMENT}]-(:{schema.CHUNK_LABEL})<-[:{schema.REL_FROM_CHUNK}]-(x)
WHERE x:Issue OR x:Decision OR x:Task OR x:Deadline
OPTIONAL MATCH (x)-[r:DECIDED_BY|ASSIGNED_TO|RAISED_BY]->(xp:Person)
OPTIONAL MATCH (x)-[:GOVERNED_BY]->(g:Clause)<-[:HAS_CLAUSE]-(:Section)<-[:HAS_SECTION]-(gd:SOPDocument)
OPTIONAL MATCH (x)-[:RESOLVES]->(iss:Issue)
WITH e, p, po, recipients, s, cons, car, brk, th, cites, x,
     collect(DISTINCT type(r) + ' ' + xp.name) AS people,
     collect(DISTINCT {{id: g.id, doc: gd.id, number: g.number, title: g.title, text: g.text}}) AS governed,
     collect(DISTINCT iss.summary) AS resolves
RETURN e.id AS email, e.date AS date, e.subject AS subject, p.name AS sender, po.name AS sender_org, recipients,
       th.id AS thread,
       CASE WHEN s IS NULL THEN NULL ELSE {{id: s.id, hbl: s.hbl, mbl: s.mbl, container: s.container, vessel: s.vessel,
            voyage: s.voyage, eta: s.eta, free_time_days: s.free_time_days, cargo: s.cargo,
            consignee: cons.name, carrier: car.name, broker: brk.name}} END AS shipment,
       cites,
       collect(DISTINCT CASE WHEN x IS NULL THEN NULL ELSE {{label: labels(x)[0], id: x.id, summary: x.summary,
            kind: x.kind, date: coalesce(x.date, x.detected_date, x.due), status: x.status, what: x.what,
            people: people, governed: governed, resolves: resolves}} END) AS extracted
"""

THREAD_NEIGHBOURS = f"""
UNWIND $emails AS eid
MATCH (e:Email {{id: eid}})-[:IN_THREAD]->(th:Thread)<-[:IN_THREAD]-(o:Email)<-[:SENT]-(p:Person)
WHERE o.id <> eid
RETURN DISTINCT th.id AS thread, o.id AS email, o.date AS date, p.name AS sender, left(o.body, 500) AS body
ORDER BY thread, date
"""


def fmt_clause(c: dict) -> str:
    return f"[{c['doc']} {c['number']}] {c['title']}: {c['text']}"


ID_RE = re.compile(r"\b(MFCIM26\d{4}|(?:HMLU|OCLU|PSSU)26\d{4}|(?:HMLU|OCLU|PSSU)\d{7}|S-\d{3})\b")

ANCHOR = """
UNWIND $ids AS x
MATCH (s:Shipment) WHERE s.hbl = x OR s.mbl = x OR s.container = x OR s.id = x
OPTIONAL MATCH (s)-[:CONSIGNEE]->(cons:Organization)
OPTIONAL MATCH (s)-[:CARRIED_BY]->(car:Organization)
OPTIONAL MATCH (s)-[:CLEARED_BY]->(brk:Organization)
OPTIONAL MATCH (n)-[:CONCERNS]->(s) WHERE n:Issue OR n:Decision OR n:Task
OPTIONAL MATCH (n)-[r:DECIDED_BY|ASSIGNED_TO|RAISED_BY]->(p:Person)
OPTIONAL MATCH (n)-[:GOVERNED_BY]->(g:Clause)<-[:HAS_CLAUSE]-(:Section)<-[:HAS_SECTION]-(gd:SOPDocument)
OPTIONAL MATCH (n)-[:FROM_CHUNK]->(:Chunk)-[:FROM_DOCUMENT]->(ev:Email)
WITH s, cons, car, brk, n, collect(DISTINCT type(r) + ' ' + p.name) AS people,
     collect(DISTINCT {id: g.id, doc: gd.id, number: g.number, title: g.title, text: g.text}) AS governed,
     collect(DISTINCT ev.id) AS evidence
RETURN s.id AS id, s.hbl AS hbl, s.mbl AS mbl, s.container AS container, s.vessel AS vessel, s.voyage AS voyage,
       s.eta AS eta, s.free_time_days AS free_time_days, s.cargo AS cargo, s.hs_code AS hs_code,
       cons.name AS consignee, car.name AS carrier, brk.name AS broker,
       collect(DISTINCT CASE WHEN n IS NULL THEN NULL ELSE {label: labels(n)[0], id: n.id, summary: n.summary,
            kind: n.kind, date: coalesce(n.date, n.detected_date, n.due), status: n.status,
            people: people, governed: governed, evidence: evidence} END) AS items
"""


def fulltext_hits(conn, question: str, k: int = TOP_K) -> list[dict]:
    q = re.sub(r"[^\w\s\-./,]", " ", question)   # keep identifiers, drop query-syntax characters
    return run(conn, f"""
        MATCH (c:{schema.CHUNK_LABEL})
        SEARCH c IN (FULLTEXT INDEX chunk_text FOR $q LIMIT $k) SCORE AS score
        MATCH (c)-[:{schema.REL_FROM_DOCUMENT}]->(e:Email)<-[:SENT]-(p:Person)
        RETURN c.id AS chunk, e.id AS email, e.date AS date, p.name AS sender, e.subject AS subject,
               c.{schema.CHUNK_TEXT_PROP} AS text, score ORDER BY score DESC""", {"q": q, "k": k})


def hybrid_seeds(conn, question: str, k: int = TOP_K) -> list[dict]:
    """Reciprocal-rank fusion of vector and full-text hits (Relate: SEARCH on both index kinds)."""
    fused: dict[str, float] = {}
    hits: dict[str, dict] = {}
    for ranked in (vector_hits(conn, question, k), fulltext_hits(conn, question, k)):
        for rank, h in enumerate(ranked):
            fused[h["email"]] = fused.get(h["email"], 0.0) + 1.0 / (60 + rank)
            hits.setdefault(h["email"], h)
    return [hits[e] for e, _ in sorted(fused.items(), key=lambda kv: -kv[1])[: k + 2]]


def fmt_item(x: dict) -> str:
    line = f"- {x['label']} {x['id']}: {x.get('summary') or x.get('what')}"
    if x.get("kind"):
        line += f" (kind {x['kind']})"
    if x.get("date"):
        line += f" (date {x['date']})"
    if x.get("status"):
        line += f" (status {x['status']})"
    if x.get("people"):
        line += " | " + "; ".join(p for p in x["people"] if p)
    if x.get("resolves"):
        line += " | resolves: " + "; ".join(x["resolves"])
    gov = [g for g in x.get("governed") or [] if g and g.get("id")]
    if gov:
        line += " | governed by " + ", ".join(f"{g['doc']} {g['number']}" for g in gov)
    if x.get("evidence"):
        line += " | evidence " + ", ".join(f"[{e}]" for e in x["evidence"] if e)
    return line


def retrieve_graph(conn, question: str) -> tuple[str, list[str], None]:
    hits = hybrid_seeds(conn, question)
    emails = [h["email"] for h in hits]
    rows = run(conn, EXPAND, {"emails": emails})
    text_by_email = {h["email"]: h["text"] for h in hits}
    parts = []
    clauses_seen: dict[str, dict] = {}
    # identifier anchors: the shipment card plus everything extracted about that shipment
    ids = sorted(set(ID_RE.findall(question)))
    if ids:
        for a in run(conn, ANCHOR, {"ids": ids}):
            block = [f"### Shipment {a['id']} (anchored by the question): HBL {a['hbl']}, MBL {a['mbl']}, container {a['container']}, "
                     f"{a['vessel']} {a['voyage']}, ETA {a['eta']}, free time {a['free_time_days']} days, cargo {a['cargo']} (HS {a['hs_code']}), "
                     f"consignee {a['consignee']}, carrier {a['carrier']}, broker {a['broker']}"]
            for x in a["items"]:
                if x:
                    block.append(fmt_item(x))
                    for g in x.get("governed") or []:
                        if g and g.get("id"):
                            clauses_seen[g["id"]] = g
            parts.append("\n".join(block))
    for r in rows:
        block = [f"### Mail [{r['email']}] {r['date'][:10]} | thread {r['thread']} | from {r['sender']} ({r['sender_org']}) "
                 f"to {', '.join(r['recipients'])}", text_by_email.get(r["email"], r["subject"])]
        if r["shipment"]:
            s = r["shipment"]
            block.append(f"Shipment {s['id']}: HBL {s['hbl']}, MBL {s['mbl']}, container {s['container']}, {s['vessel']} {s['voyage']}, "
                         f"ETA {s['eta']}, free time {s['free_time_days']} days, cargo {s['cargo']}, consignee {s['consignee']}, "
                         f"carrier {s['carrier']}, broker {s['broker']}")
        for x in r["extracted"]:
            if not x:
                continue
            block.append(fmt_item(x))
            for g in x.get("governed") or []:
                if g and g.get("id"):
                    clauses_seen[g["id"]] = g
        for c in r["cites"]:
            if c and c.get("id"):
                clauses_seen[c["id"]] = c
                block.append(f"- cites {c['doc']} {c['number']}")
        parts.append("\n".join(block))
    neigh = run(conn, THREAD_NEIGHBOURS, {"emails": emails})
    if neigh:
        parts.append("### Other mails in the same threads (excerpts)\n" + "\n".join(
            f"[{n['email']}] {n['date'][:10]} {n['sender']} ({n['thread']}): {n['body'].strip().splitlines()[0] if n['body'].strip() else ''} ..."
            for n in neigh))
    if clauses_seen:
        parts.append("### SOP clauses linked to the above\n" + "\n".join(fmt_clause(c) for c in clauses_seen.values()))
    # the SOPs are a second corpus: search the clauses directly too (vector + full-text), for rule questions
    direct = [c for c in clause_hits(conn, question) if c["id"] not in clauses_seen]
    if direct:
        parts.append("### SOP clauses matching the question\n" + "\n".join(fmt_clause(c) for c in direct))
    retrieved = emails + [n["email"] for n in neigh] + list(clauses_seen) + [c["id"] for c in direct]
    return "\n\n".join(parts), retrieved, None


def clause_hits(conn, question: str, k: int = 3) -> list[dict]:
    """Top clauses by RRF of vector and full-text search over Clause nodes."""
    q = re.sub(r"[^\w\s\-./,]", " ", question)
    vec = run(conn, f"""
        MATCH (c:Clause) SEARCH c IN (VECTOR INDEX clause_embedding FOR vector($v, {schema.EMBEDDING_DIM}, FLOAT32) LIMIT $k) SCORE AS s
        MATCH (d:SOPDocument)-[:HAS_SECTION]->(:Section)-[:HAS_CLAUSE]->(c)
        RETURN c.id AS id, d.id AS doc, c.number AS number, c.title AS title, c.text AS text ORDER BY s DESC""",
        {"v": embed(question), "k": k})
    ft = run(conn, """
        MATCH (c:Clause) SEARCH c IN (FULLTEXT INDEX clause_text FOR $q LIMIT $k) SCORE AS s
        MATCH (d:SOPDocument)-[:HAS_SECTION]->(:Section)-[:HAS_CLAUSE]->(c)
        RETURN c.id AS id, d.id AS doc, c.number AS number, c.title AS title, c.text AS text ORDER BY s DESC""",
        {"q": q, "k": k})
    fused: dict[str, float] = {}
    seen: dict[str, dict] = {}
    for ranked in (vec, ft):
        for rank, c in enumerate(ranked):
            fused[c["id"]] = fused.get(c["id"], 0.0) + 1.0 / (60 + rank)
            seen.setdefault(c["id"], c)
    return [seen[i] for i, _ in sorted(fused.items(), key=lambda kv: -kv[1])[:k]]


# --------------------------------------------------------------------------- 3. text2cypher
GRAPH_SCHEMA = f"""
Node labels and properties:
  Person {{id, name, email, title, is_internal}}
  Organization {{id, name, type}}            type in forwarder|carrier|broker|consignee|shipper|trucker|terminal
  Shipment {{id (S-001..S-025), hbl (MFCIM26xxxx), mbl, container, container_type (20GP|40HC|40RF), vessel, voyage,
            pol, pol_name, pod, eta (YYYY-MM-DD string), free_time_days, cargo, hs_code}}
  Thread {{id, subject}}
  Email {{id (M-xxxx), message_id, subject, date (ISO datetime string), body}}
  Attachment {{id, filename}}
  Chunk {{id, text, embedding}}                     one chunk per e-mail body
  Issue {{id, summary, kind, shipment_id, detected_date, sop_refs}}   kind in {schema.ISSUE_KINDS}
  Decision {{id, summary, date, shipment_id, sop_refs}}
  Task {{id, summary, due, status (open|done|duplicate - a restatement of another thread's task, see DUPLICATE_OF), shipment_id}}
  Deadline {{id, date, what, shipment_id}}
  SOPDocument {{id (SOP-01..SOP-05), title, version, effective_date, filename}}
  Section {{id, number, title}}
  Clause {{id (e.g. SOP-03-C4.2), number (e.g. 4.2), title, text, embedding}}
Relationships:
  (Person)-[:WORKS_AT]->(Organization)
  (Person)-[:SENT]->(Email), (Email)-[:TO]->(Person), (Email)-[:CC]->(Person)
  (Email)-[:IN_THREAD]->(Thread), (Email)-[:REPLIES_TO]->(Email), (Email)-[:ABOUT]->(Shipment), (Email)-[:HAS_ATTACHMENT]->(Attachment)
  (Shipment)-[:CARRIED_BY]->(Organization), (Shipment)-[:CLEARED_BY]->(Organization), (Shipment)-[:CONSIGNEE]->(Organization), (Shipment)-[:SHIPPER]->(Organization)
  (Chunk)-[:FROM_DOCUMENT]->(Email)
  (Issue|Decision|Task|Deadline)-[:FROM_CHUNK]->(Chunk)        provenance: which mail the item was extracted from
  (Issue|Decision|Task|Deadline)-[:CONCERNS]->(Shipment)
  (Decision)-[:DECIDED_BY]->(Person), (Task)-[:ASSIGNED_TO]->(Person), (Issue)-[:RAISED_BY]->(Person)
  (Decision)-[:RESOLVES]->(Issue), (Task)-[:HAS_DEADLINE]->(Deadline), (Task)-[:DUPLICATE_OF]->(Task)
  (SOPDocument)-[:HAS_SECTION]->(Section), (Section)-[:HAS_CLAUSE]->(Clause), (Section)-[:NEXT_SECTION]->(Section), (Clause)-[:NEXT_CLAUSE]->(Clause)
  (Issue|Decision)-[:GOVERNED_BY]->(Clause), (Email)-[:CITES]->(Clause)
Dialect: openCypher / Cypher 25 on Relate (embedded Neo4j-compatible). No APOC, no GDS. Dates are strings: compare with < > or use date(substring(e.date,0,10)).
Full-text style matching: use toLower(x) CONTAINS 'word'. Always RETURN explicit property values, not whole nodes. Add LIMIT 50.
"""

T2C_SYSTEM = f"""You translate questions about a freight forwarder's mailbox + SOP knowledge graph into ONE read-only Cypher query.
{GRAPH_SCHEMA}
Return only the Cypher, no prose, no code fences. Prefer returning the human-readable fields (names, summaries, dates, hbl) the
answer needs. For "who decided / who owns" questions traverse DECIDED_BY / ASSIGNED_TO. For compliance questions return the
Issue/Decision summary together with the GOVERNED_BY clause number, title and text. For "what is open" use Task.status = 'open'.
When the question names an HBL or MBL, match Shipment by that property and reach mails via (Email)-[:ABOUT]->(Shipment) and
extracted items via CONCERNS."""


def gen_cypher(client: anthropic.Anthropic, question: str, error: str | None = None, previous: str | None = None) -> tuple[str, dict]:
    content = f"Question: {question}"
    if error:
        content += f"\n\nYour previous query failed:\n{previous}\n\nError: {error}\n\nReturn a corrected query."
    # max_tokens must leave room for adaptive thinking; 800 truncated queries mid-clause in the first eval run
    resp = client.messages.create(model=schema.LLM_MODEL, max_tokens=4000, system=T2C_SYSTEM,
                                  messages=[{"role": "user", "content": content}])
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    text = re.sub(r"^```(?:cypher)?\s*|\s*```$", "", text, flags=re.I | re.M).strip()
    if resp.stop_reason == "max_tokens":
        text += "\n// TRUNCATED: generation hit max_tokens"
    return text, {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}


def retrieve_text2cypher(conn, client: anthropic.Anthropic, question: str) -> tuple[str, list[str], str, dict, str | None]:
    usage = {"input": 0, "output": 0}
    cypher, u = gen_cypher(client, question)
    usage = {k: usage[k] + u[k] for k in usage}
    err = None
    for attempt in range(2):
        try:
            rows = run(conn, cypher)
            err = None
            break
        except Exception as e:  # noqa: BLE001 - surface the DB error to the model once
            err = str(e)[:600]
            if attempt == 0:
                cypher, u = gen_cypher(client, question, err, cypher)
                usage = {k: usage[k] + u[k] for k in usage}
    if err:
        return f"Cypher failed: {err}", [], cypher, usage, err
    rows = rows[:50]
    ctx = f"Cypher:\n{cypher}\n\nRows ({len(rows)}):\n" + "\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in rows)
    ids = sorted({v for r in rows for v in r.values() if isinstance(v, str) and re.fullmatch(r"(M-\d{4}|S-\d{3}|SOP-0\d-C[\d.]+|TH-\d{4}-\w+)", v)})
    return ctx, ids, cypher, usage, None


# --------------------------------------------------------------------------- entry
def ask(conn, client: anthropic.Anthropic, question: str, mode: str) -> Result:
    if mode == "vector":
        ctx, ids, cypher = retrieve_vector(conn, question)
        usage = {"input": 0, "output": 0}
        err = None
    elif mode == "graph":
        ctx, ids, cypher = retrieve_graph(conn, question)
        usage = {"input": 0, "output": 0}
        err = None
    elif mode == "text2cypher":
        ctx, ids, cypher, usage, err = retrieve_text2cypher(conn, client, question)
    else:
        raise ValueError(mode)
    ans, u = answer(client, question, ctx if len(ctx) < 60000 else ctx[:60000])
    usage = {k: usage[k] + u[k] for k in usage}
    return Result(mode, question, ans, ctx, cypher, ids, usage, err)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--mode", choices=MODES + ("all",), default="all")
    ap.add_argument("--show-context", action="store_true")
    args = ap.parse_args()
    conn = connect(read_only=True)
    client = anthropic.Anthropic()
    for mode in (MODES if args.mode == "all" else (args.mode,)):
        r = ask(conn, client, args.question, mode)
        print(f"\n===== {mode} =====")
        if r.cypher:
            print("Cypher:", r.cypher)
        if args.show_context:
            print("--- context ---\n", r.context[:4000], "\n--- end ---")
        print(r.answer)
    conn.close()


if __name__ == "__main__":
    main()
