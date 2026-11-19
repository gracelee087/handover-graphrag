"""Step 2: deterministic structure layer -> Relate store (no LLM).

What the successor realistically has on day one:
  * the company directory (people, organizations)            -> world.yaml people/organizations
  * the TMS shipment list (B/L, container, vessel, ETA...)    -> world.yaml shipments (operational fields only)
  * the predecessor's mailbox                                 -> data/mail/eml/*.eml, parsed with the stdlib
Incidents, decisions, tasks and SOP clauses are NOT loaded here - they have to
be recovered from mail bodies (step 3a) and SOP PDFs (step 3b).

Email -> Shipment links come from HBL/MBL numbers found in subject or body.

Run:  uv run python src/load_structure.py
"""
from __future__ import annotations

import re
import sys
from email import policy
from email.parser import BytesParser
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import schema  # noqa: E402
from store import connect, counts, init_schema, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
W = yaml.safe_load((ROOT / "world" / "world.yaml").read_text(encoding="utf-8"))
EML_DIR = ROOT / "data" / "mail" / "eml"

SHIPMENT_PROPS = ["id", "mbl", "hbl", "container", "container_type", "vessel", "voyage",
                  "pol", "pol_name", "pod", "pod_name", "eta", "free_time_days", "cargo", "hs_code"]
HBL_RE = re.compile(r"\bMFCIM26\d{4}\b")
MBL_RE = re.compile(r"\b(?:HMLU|OCLU|PSSU)26\d{4}\b")


def load_directory(conn) -> None:
    run(conn, "UNWIND $rows AS r MERGE (o:Organization {id: r.id}) "
              "SET o.name = r.name, o.type = r.type, o.domain = r.domain",
        {"rows": W["organizations"]})
    run(conn, "UNWIND $rows AS r MERGE (p:Person {id: r.id}) "
              "SET p.name = r.name, p.email = r.email, p.title = r.title, p.is_internal = r.is_internal "
              "WITH p, r MATCH (o:Organization {id: r.org}) MERGE (p)-[:WORKS_AT]->(o)",
        {"rows": W["people"]})


def load_shipments(conn) -> None:
    rows = []
    for s in W["shipments"]:
        row = {k: s[k] for k in SHIPMENT_PROPS}
        row.update({"carrier": s["carrier"], "broker": s["broker"], "consignee": s["consignee"],
                    "shipper": s.get("shipper")})
        rows.append(row)
    sets = ", ".join(f"s.{k} = r.{k}" for k in SHIPMENT_PROPS if k != "id")
    run(conn, f"UNWIND $rows AS r MERGE (s:Shipment {{id: r.id}}) SET {sets}", {"rows": rows})
    for rel, key in [("CARRIED_BY", "carrier"), ("CLEARED_BY", "broker"), ("CONSIGNEE", "consignee"),
                     ("SHIPPER", "shipper")]:
        run(conn, f"UNWIND $rows AS r WITH r WHERE r.{key} IS NOT NULL "
                  f"MATCH (s:Shipment {{id: r.id}}), (o:Organization {{id: r.{key}}}) MERGE (s)-[:{rel}]->(o)",
            {"rows": rows})


def parse_eml(path: Path) -> dict:
    msg = BytesParser(policy=policy.default).parse(path.open("rb"))
    body_part = msg.get_body(preferencelist=("plain",))
    body = body_part.get_content() if body_part else ""
    refs = (msg.get("References") or "").split()
    return {
        "id": msg["Message-ID"].strip("<>").split("@")[0],
        "message_id": msg["Message-ID"].strip(),
        "date": parsedate_to_datetime(msg["Date"]).isoformat(),
        "subject": msg["Subject"].strip(),
        "body": body,
        "from": getaddresses([msg["From"]])[0][1].lower(),
        "to": [a.lower() for _, a in getaddresses(msg.get_all("To", []))],
        "cc": [a.lower() for _, a in getaddresses(msg.get_all("Cc", []))],
        "in_reply_to": (msg.get("In-Reply-To") or "").strip() or None,
        "root": refs[0] if refs else msg["Message-ID"].strip(),
        "attachments": [p.get_filename() for p in msg.iter_attachments() if p.get_filename()],
    }


def load_mail(conn) -> dict:
    mails = sorted((parse_eml(p) for p in EML_DIR.glob("*.eml")), key=lambda m: (m["date"], m["id"]))
    by_mid = {m["message_id"]: m for m in mails}
    hbl = {s["hbl"]: s["id"] for s in W["shipments"]}
    mbl = {s["mbl"]: s["id"] for s in W["shipments"]}
    thread_ids: dict[str, str] = {}
    for m in mails:
        root = by_mid.get(m["root"], m)
        thread_ids.setdefault(root["message_id"], f"TH-{root['id'][2:]}")  # TH-0001 from M-0001
        m["thread"] = thread_ids[root["message_id"]]
        m["thread_subject"] = re.sub(r"^(re|fw|fwd):\s*", "", root["subject"], flags=re.I)
        text = m["subject"] + "\n" + m["body"]
        m["shipments"] = sorted({hbl[h] for h in HBL_RE.findall(text) if h in hbl}
                                | {mbl[x] for x in MBL_RE.findall(text) if x in mbl})

    run(conn, "UNWIND $rows AS r MERGE (t:Thread {id: r.thread}) SET t.subject = r.thread_subject", {"rows": mails})
    run(conn, "UNWIND $rows AS r MERGE (e:Email {id: r.id}) "
              "SET e.message_id = r.message_id, e.subject = r.subject, e.date = r.date, e.body = r.body "
              "WITH e, r MATCH (t:Thread {id: r.thread}) MERGE (e)-[:IN_THREAD]->(t)", {"rows": mails})
    run(conn, "UNWIND $rows AS r MATCH (e:Email {id: r.id}), (p:Person {email: r.from}) MERGE (p)-[:SENT]->(e)",
        {"rows": mails})
    for rel, key in [("TO", "to"), ("CC", "cc")]:
        run(conn, f"UNWIND $rows AS r UNWIND r.{key} AS addr "
                  f"MATCH (e:Email {{id: r.id}}), (p:Person {{email: addr}}) MERGE (e)-[:{rel}]->(p)", {"rows": mails})
    run(conn, "UNWIND $rows AS r WITH r WHERE r.in_reply_to IS NOT NULL "
              "MATCH (e:Email {id: r.id}), (parent:Email {message_id: r.in_reply_to}) MERGE (e)-[:REPLIES_TO]->(parent)",
        {"rows": mails})
    run(conn, "UNWIND $rows AS r UNWIND r.shipments AS sid "
              "MATCH (e:Email {id: r.id}), (s:Shipment {id: sid}) MERGE (e)-[:ABOUT]->(s)", {"rows": mails})
    run(conn, "UNWIND $rows AS r UNWIND r.attachments AS fn "
              "MATCH (e:Email {id: r.id}) MERGE (a:Attachment {id: r.id + '/' + fn}) SET a.filename = fn "
              "MERGE (e)-[:HAS_ATTACHMENT]->(a)", {"rows": mails})
    return {"mails": len(mails), "threads": len(thread_ids),
            "mails_without_shipment": sum(1 for m in mails if not m["shipments"]),
            "unknown_senders": sorted({m["from"] for m in mails} - {p["email"] for p in W["people"]})}


def main() -> None:
    conn = connect()
    init_schema(conn)
    load_directory(conn)
    load_shipments(conn)
    stats = load_mail(conn)
    print("mail stats:", stats)
    print("node counts:", counts(conn))
    rels = run(conn, "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY t")
    print("rel counts:", {r["t"]: r["c"] for r in rels})
    conn.close()


if __name__ == "__main__":
    main()
