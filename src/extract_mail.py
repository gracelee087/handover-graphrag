"""Step 3a: mail bodies -> lexical graph (Chunk + embedding) and extracted entities.

Per thread (so the model sees the whole conversation) Claude extracts
Issue / Decision / Task / Deadline nodes with structured output, constrained to
the vocabulary in schema.py. People are resolved to the directory by name,
shipments by id (the prompt lists the HBL/MBL -> S-id mapping for the thread).
Every extracted node keeps provenance: (node)-[:FROM_CHUNK]->(Chunk)-[:FROM_DOCUMENT]->(Email).

neo4j-graphrag's SimpleKGPipeline is not used because its Neo4jWriter needs
APOC (apoc.merge.relationship), which the embedded Relate store does not have.

Outputs: data/mail/extract/TH-xxx.json (cache), data/mail/_extract_usage.json
Run:  uv run python src/extract_mail.py [--limit N] [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import anthropic
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import schema  # noqa: E402
from store import connect, counts, init_schema, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)
W = yaml.safe_load((ROOT / "world" / "world.yaml").read_text(encoding="utf-8"))
CACHE = ROOT / "data" / "mail" / "extract"
PRICE_IN, PRICE_OUT = 4.0, 20.0
HBL_RE = re.compile(r"\bMFCIM26\d{4}\b")
MBL_RE = re.compile(r"\b(?:HMLU|OCLU|PSSU)26\d{4}\b")
SID_RE = re.compile(r"\bS-\d{3}\b")

Kind = Literal[tuple(schema.ISSUE_KINDS)]  # type: ignore[valid-type]


class Issue(BaseModel):
    key: str
    summary: str
    kind: Kind
    shipment_id: str | None = None
    raised_by: str | None = None
    detected_date: str | None = None
    sop_refs: list[str] = []
    evidence: list[str]


class Decision(BaseModel):
    key: str
    summary: str
    date: str
    decided_by: str
    shipment_id: str | None = None
    resolves: str | None = None
    sop_refs: list[str] = []
    evidence: list[str]


class Task(BaseModel):
    key: str
    summary: str
    owner: str
    due: str | None = None
    status: Literal["open", "done"]
    shipment_id: str | None = None
    evidence: list[str]


class Deadline(BaseModel):
    date: str
    what: str
    shipment_id: str | None = None
    evidence: list[str]


class ThreadExtraction(BaseModel):
    issues: list[Issue]
    decisions: list[Decision]
    tasks: list[Task]
    deadlines: list[Deadline]


SYSTEM = f"""You are building a knowledge graph from the mailbox of an import operator at {W['meta']['company']} (sea-freight forwarder, Busan).
Read one e-mail thread and extract:
- issues: problems/exceptions with a shipment (vessel delay, missing documents, demurrage, detention, customs inspection, HS code dispute, D/O or release hold, cargo damage claim, urgent clearance, manifest discrepancy). kind must be one of {schema.ISSUE_KINDS}. One issue per distinct problem, not per mail.
- decisions: explicit business decisions (who decided what, on which date). Include who absorbs a cost, approvals, policy changes, handovers of responsibility. decided_by is the person who made the decision (not who reported it).
- tasks: action items that a mail explicitly assigns to a named person AND that are still outstanding at the end of the thread (status "open"), or that the thread explicitly reports as completed (status "done"). Do NOT extract as tasks: routine closing steps that follow automatically from a document (issuing an invoice from a cost sheet, filing a POD or declaration), standard operational steps that are part of every file (arranging pick-up, requesting a D/O, sending an arrival notice, telex release) unless a mail hands them over as an open item with a named owner, requests to vendors that the vendor fulfils later in the thread, or anything the same thread shows was done. due is the date written in the mail (YYYY-MM-DD) or null.
- deadlines: dated cut-offs mentioned (free-time expiry, due dates, ETAs are NOT deadlines).
Rules: use the exact dates written in the mails (YYYY-MM-DD); summaries are one sentence with concrete facts (amounts, counts, container/B/L numbers). Names must be copied exactly as they appear in the participant list. shipment_id must come from the mapping given (or null). sop_refs are SOP clause citations exactly as written, e.g. "SOP-03 section 4.2" (empty list if none). evidence lists the message ids (M-xxxx) that support the item. keys are short unique labels like ISS1, DEC1, TASK1. Routine operations (sending an arrival notice, booking a truck, confirming receipt) are NOT issues or decisions; a routine request to a vendor is NOT a task unless the thread treats it as an open action item. Only extract an issue when the mails explicitly describe a problem; never infer a problem from something that is merely not mentioned. Return empty lists when a thread has nothing to extract."""


def thread_prompt(th_id: str, mails: list[dict], people_by_email: dict) -> tuple[str, dict]:
    text = "\n".join(m["subject"] + "\n" + m["body"] for m in mails)
    hbl = {s["hbl"]: s["id"] for s in W["shipments"]}
    mbl = {s["mbl"]: s["id"] for s in W["shipments"]}
    ship_map = {h: hbl[h] for h in HBL_RE.findall(text) if h in hbl}
    ship_map.update({x: mbl[x] for x in MBL_RE.findall(text) if x in mbl})
    sids = {s["id"] for s in W["shipments"]}
    ship_map.update({x: x for x in SID_RE.findall(text) if x in sids})   # mails may name a file by its S-id only
    names = sorted({people_by_email[a]["name"] for m in mails for a in [m["from"], *m["to"], *m["cc"]] if a in people_by_email})
    lines = [f"Thread {th_id}", "Participants: " + ", ".join(names),
             "Shipment mapping (B/L number or S-id -> shipment_id): " + (", ".join(f"{k} -> {v}" for k, v in ship_map.items()) or "none"), ""]
    for m in mails:
        to = ", ".join(people_by_email[a]["name"] for a in m["to"] if a in people_by_email)
        lines += [f"=== {m['id']} | {m['date'][:10]} | from {people_by_email[m['from']]['name']} | to {to}",
                  f"Subject: {m['subject']}", m["body"].strip(), ""]
    return "\n".join(lines), ship_map


def extract_thread(client: anthropic.Anthropic, th_id: str, mails: list[dict], people_by_email: dict) -> dict:
    cache = CACHE / f"{th_id}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    prompt, _ = thread_prompt(th_id, mails, people_by_email)
    resp = client.messages.parse(model=schema.LLM_MODEL, max_tokens=8000, system=SYSTEM,
                                 messages=[{"role": "user", "content": prompt}], output_format=ThreadExtraction)
    rec = {"thread": th_id, "usage": {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens},
           "extraction": resp.parsed_output.model_dump()}
    cache.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec


def load_mailbox(conn) -> dict[str, list[dict]]:
    rows = run(conn, "MATCH (p:Person)-[:SENT]->(e:Email)-[:IN_THREAD]->(t:Thread) "
                     "OPTIONAL MATCH (e)-[:TO]->(to:Person) WITH p, e, t, collect(to.email) AS to "
                     "OPTIONAL MATCH (e)-[:CC]->(cc:Person) WITH p, e, t, to, collect(cc.email) AS cc "
                     "RETURN t.id AS thread, e.id AS id, e.date AS date, e.subject AS subject, e.body AS body, "
                     "p.email AS `from`, to, cc ORDER BY e.date, e.id")
    threads: dict[str, list[dict]] = {}
    for r in rows:
        threads.setdefault(r["thread"], []).append(r)
    return threads


def write_chunks(conn, threads: dict[str, list[dict]], embedder: SentenceTransformer) -> int:
    rows = [{"id": f"{m['id']}-c0", "email": m["id"], "index": 0, "text": f"Subject: {m['subject']}\n\n{m['body'].strip()}"}
            for mails in threads.values() for m in mails]
    vecs = embedder.encode([r["text"] for r in rows], normalize_embeddings=True, batch_size=64).tolist()
    for r, v in zip(rows, vecs):
        r["emb"] = v
    for i in range(0, len(rows), 100):
        run(conn, f"UNWIND $rows AS r MATCH (e:Email {{id: r.email}}) MERGE (c:{schema.CHUNK_LABEL} {{id: r.id}}) "
                  f"SET c.index = r.index, c.{schema.CHUNK_TEXT_PROP} = r.text, "
                  f"c.{schema.CHUNK_EMBEDDING_PROP} = vector(r.emb, {schema.EMBEDDING_DIM}, FLOAT32) "
                  f"MERGE (c)-[:{schema.REL_FROM_DOCUMENT}]->(e)", {"rows": rows[i:i + 100]})
    return len(rows)


def write_extraction(conn, th_id: str, ex: dict, name_to_pid: dict[str, str], stats: dict) -> None:
    def pid(name: str | None) -> str | None:
        if not name:
            return None
        p = name_to_pid.get(name.strip().lower())
        if p is None:
            stats["unresolved_people"].add(name)
        return p

    def node(label: str, key: str, props: dict, person_rels: list[tuple[str, str | None]], evidence: list[str]) -> str:
        nid = f"{th_id}-{key}"
        sets = ", ".join(f"n.{k} = ${k}" for k in props)
        run(conn, f"MERGE (n:{label} {{id: $id}}) SET {sets}", {"id": nid, **props})
        if props.get("shipment_id"):
            n = run(conn, f"MATCH (n:{label} {{id: $id}}), (s:Shipment {{id: $sid}}) MERGE (n)-[:CONCERNS]->(s) RETURN count(*) AS c",
                    {"id": nid, "sid": props["shipment_id"]})
            if not n or n[0]["c"] == 0:
                stats["bad_shipment_ids"].add(props["shipment_id"])
        for rel, p in person_rels:
            if p:
                run(conn, f"MATCH (n:{label} {{id: $id}}), (p:Person {{id: $pid}}) MERGE (n)-[:{rel}]->(p)", {"id": nid, "pid": p})
        for mid in evidence:
            r = run(conn, f"MATCH (n:{label} {{id: $id}}), (c:{schema.CHUNK_LABEL} {{id: $cid}}) "
                          f"MERGE (n)-[:{schema.REL_FROM_CHUNK}]->(c) RETURN count(*) AS c", {"id": nid, "cid": f"{mid}-c0"})
            if not r or r[0]["c"] == 0:
                stats["bad_evidence"].add(mid)
        stats["nodes"][label] = stats["nodes"].get(label, 0) + 1
        return nid

    for it in ex["issues"]:
        node("Issue", it["key"], {"summary": it["summary"], "kind": it["kind"], "shipment_id": it["shipment_id"],
                                 "detected_date": it["detected_date"], "sop_refs": it["sop_refs"]},
             [("RAISED_BY", pid(it["raised_by"]))], it["evidence"])
    for d in ex["decisions"]:
        nid = node("Decision", d["key"], {"summary": d["summary"], "date": d["date"], "shipment_id": d["shipment_id"],
                                         "sop_refs": d["sop_refs"]},
                   [("DECIDED_BY", pid(d["decided_by"]))], d["evidence"])
        if d.get("resolves"):
            run(conn, "MATCH (d:Decision {id: $d}), (i:Issue {id: $i}) MERGE (d)-[:RESOLVES]->(i)",
                {"d": nid, "i": f"{th_id}-{d['resolves']}"})
    for t in ex["tasks"]:
        nid = node("Task", t["key"], {"summary": t["summary"], "due": t["due"], "status": t["status"],
                                     "shipment_id": t["shipment_id"]},
                   [("ASSIGNED_TO", pid(t["owner"]))], t["evidence"])
        if t.get("due"):
            run(conn, "MATCH (t:Task {id: $t}) MERGE (dl:Deadline {id: $t + '-due'}) SET dl.date = $date, dl.what = $what "
                      "MERGE (t)-[:HAS_DEADLINE]->(dl)", {"t": nid, "date": t["due"], "what": "due date: " + t["summary"]})
    for i, dl in enumerate(ex["deadlines"], 1):
        node("Deadline", f"DL{i}", {"date": dl["date"], "what": dl["what"], "shipment_id": dl["shipment_id"]}, [], dl["evidence"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    conn = connect()
    init_schema(conn)
    threads = load_mailbox(conn)
    if args.limit:
        threads = dict(list(threads.items())[: args.limit])
    people_by_email = {p["email"]: p for p in W["people"]}
    name_to_pid = {p["name"].lower(): p["id"] for p in W["people"]}

    embedder = SentenceTransformer(schema.EMBEDDING_MODEL)
    n_chunks = write_chunks(conn, threads, embedder)
    print(f"chunks written: {n_chunks}", flush=True)

    client = anthropic.Anthropic()
    recs: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract_thread, client, th, mails, people_by_email): th for th, mails in threads.items()}
        for f in as_completed(futs):
            rec = f.result()
            recs[rec["thread"]] = rec
            e = rec["extraction"]
            print(f"{rec['thread']}: issues={len(e['issues'])} decisions={len(e['decisions'])} "
                  f"tasks={len(e['tasks'])} deadlines={len(e['deadlines'])}", flush=True)

    stats = {"nodes": {}, "unresolved_people": set(), "bad_shipment_ids": set(), "bad_evidence": set()}
    for th_id in threads:
        write_extraction(conn, th_id, recs[th_id]["extraction"], name_to_pid, stats)
    stats = {k: (sorted(v) if isinstance(v, set) else v) for k, v in stats.items()}
    tin = sum(r["usage"]["input"] for r in recs.values())
    tout = sum(r["usage"]["output"] for r in recs.values())
    usage = {"model": schema.LLM_MODEL, "threads": len(recs), "input_tokens": tin, "output_tokens": tout,
             "cost_usd_list_price": round(tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT, 3), "stats": stats}
    (ROOT / "data" / "mail" / "_extract_usage.json").write_text(json.dumps(usage, indent=1), encoding="utf-8")
    print(json.dumps(usage, indent=1))
    print("node counts:", counts(conn))
    conn.close()


if __name__ == "__main__":
    main()
