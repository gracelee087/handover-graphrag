"""Step 7: handover agent - Claude with tools over the Relate knowledge graph.

The successor asks questions in plain English; Claude decides which tool to use:
    search_mailbox   hybrid (vector + full-text) search seeded expansion through the graph
    shipment_file    everything the graph knows about one shipment (by HBL/MBL/container/id)
    open_tasks       open action items, optionally for one owner
    sop_clause       the text of an SOP clause with its neighbours
    read_mail        one e-mail in full
    run_cypher       any read-only Cypher against the schema (for counts, lists, joins)
Every answer must cite mail ids [M-0231] and clauses [SOP-03 4.2] so the successor can verify.

Run:  uv run python src/agent.py                 # interactive
      uv run python src/agent.py "question"      # one shot
      uv run python src/agent.py --demo          # scripted questions (used in the demo video)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import anthropic
from anthropic import beta_tool
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import schema  # noqa: E402
from retrievers import ANCHOR, GRAPH_SCHEMA, retrieve_graph  # noqa: E402
from store import connect, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)
CONN = connect(read_only=True)
VERBOSE = True


def log(tool: str, arg: str) -> None:
    if VERBOSE:
        print(f"   → {tool}({arg[:90]})", flush=True)


def rows_json(rows: list[dict], limit: int = 50) -> str:
    return json.dumps(rows[:limit], ensure_ascii=False, default=str, indent=0)


@beta_tool
def search_mailbox(question: str) -> str:
    """Search the predecessor's mailbox for a question. Combines vector and full-text search over the
    e-mails, then expands each hit through the graph: sender/recipients, the shipment, the issues,
    decisions and tasks extracted from that mail (with who decided / owns them) and the SOP clauses they
    are governed by. Use this first for 'what happened / who decided / why' questions.

    Args:
        question: The question or keywords to search for (include HBL/MBL numbers if known).
    """
    log("search_mailbox", question)
    ctx, _, _ = retrieve_graph(CONN, question)
    return ctx[:40000]


@beta_tool
def shipment_file(identifier: str) -> str:
    """Return the shipment card (B/L numbers, container, vessel, ETA, free time, parties) plus every
    issue, decision and task recorded for it, and the list of all e-mails about it.

    Args:
        identifier: HBL (MFCIM26xxxx), MBL, container number or shipment id (S-009).
    """
    log("shipment_file", identifier)
    cards = run(CONN, ANCHOR, {"ids": [identifier.strip()]})
    if not cards:
        return f"No shipment matches {identifier!r}."
    mails = run(CONN, """
        MATCH (s:Shipment) WHERE s.hbl = $x OR s.mbl = $x OR s.container = $x OR s.id = $x
        MATCH (e:Email)-[:ABOUT]->(s), (p:Person)-[:SENT]->(e)
        RETURN e.id AS id, left(e.date, 10) AS date, p.name AS sender, e.subject AS subject ORDER BY e.date""",
        {"x": identifier.strip()})
    return rows_json(cards, 5) + "\n\nMails about this shipment:\n" + rows_json(mails, 80)


@beta_tool
def open_tasks(owner: str = "") -> str:
    """List open tasks (action items not yet done) with owner, due date, shipment and the mail they come from.

    Args:
        owner: Optional person name to filter by (e.g. "Daniel Cho"). Empty for everyone.
    """
    log("open_tasks", owner)
    rows = run(CONN, """
        MATCH (t:Task {status: 'open'})
        OPTIONAL MATCH (t)-[:ASSIGNED_TO]->(p:Person)
        OPTIONAL MATCH (t)-[:CONCERNS]->(s:Shipment)
        OPTIONAL MATCH (t)-[:FROM_CHUNK]->(:Chunk)-[:FROM_DOCUMENT]->(e:Email)
        WITH t, p, s, collect(DISTINCT e.id) AS evidence
        WHERE $owner = '' OR toLower(p.name) CONTAINS toLower($owner)
        RETURN t.id AS id, t.summary AS summary, t.due AS due, p.name AS owner, s.hbl AS hbl, evidence
        ORDER BY coalesce(t.due, '9999'), t.id""", {"owner": owner})
    return rows_json(rows, 80)


@beta_tool
def sop_clause(reference: str) -> str:
    """Return the full text of an SOP clause and the clauses before and after it.

    Args:
        reference: e.g. "SOP-03 4.2", "SOP-03 section 4.2" or the id "SOP-03-C4.2".
    """
    log("sop_clause", reference)
    m = re.search(r"(SOP-0\d)\D*(\d+\.\d+)", reference, re.I)
    if not m:
        return "Give the reference as 'SOP-0N x.y'."
    cid = f"{m.group(1).upper()}-C{m.group(2)}"
    rows = run(CONN, """
        MATCH (d:SOPDocument)-[:HAS_SECTION]->(sec:Section)-[:HAS_CLAUSE]->(c:Clause {id: $id})
        OPTIONAL MATCH (prev:Clause)-[:NEXT_CLAUSE]->(c)
        OPTIONAL MATCH (c)-[:NEXT_CLAUSE]->(next:Clause)
        OPTIONAL MATCH (x)-[:GOVERNED_BY]->(c)
        RETURN d.id AS doc, d.title AS doc_title, d.version AS version, sec.number + ' ' + sec.title AS section,
               c.number AS number, c.title AS title, c.text AS text,
               prev.number + ' ' + prev.title AS previous_clause, next.number + ' ' + next.title AS next_clause,
               collect(DISTINCT labels(x)[0] + ' ' + x.id + ': ' + x.summary) AS cases_governed_by_this_clause""",
        {"id": cid})
    return rows_json(rows) if rows else f"No clause {cid}."


@beta_tool
def read_mail(email_id: str) -> str:
    """Return one e-mail in full (headers, body, attachments, thread neighbours).

    Args:
        email_id: The mail id, e.g. "M-0231".
    """
    log("read_mail", email_id)
    rows = run(CONN, """
        MATCH (p:Person)-[:SENT]->(e:Email {id: $id})-[:IN_THREAD]->(t:Thread)
        OPTIONAL MATCH (e)-[:TO]->(to:Person)
        OPTIONAL MATCH (e)-[:CC]->(cc:Person)
        OPTIONAL MATCH (e)-[:HAS_ATTACHMENT]->(a:Attachment)
        OPTIONAL MATCH (o:Email)-[:IN_THREAD]->(t) WHERE o.id <> e.id
        RETURN e.id AS id, e.date AS date, p.name AS sender, collect(DISTINCT to.name) AS to, collect(DISTINCT cc.name) AS cc,
               e.subject AS subject, e.body AS body, collect(DISTINCT a.filename) AS attachments,
               t.id AS thread, collect(DISTINCT o.id) AS other_mails_in_thread""", {"id": email_id.strip()})
    return rows_json(rows) if rows else f"No mail {email_id}."


@beta_tool
def run_cypher(query: str) -> str:
    """Run a read-only Cypher query against the knowledge graph (counts, lists, joins the other tools
    do not cover). Relate dialect: openCypher, no APOC. Always RETURN property values and add LIMIT.

    Args:
        query: The Cypher query.
    """
    log("run_cypher", query.replace("\n", " "))
    try:
        return rows_json(run(CONN, query))
    except Exception as e:  # noqa: BLE001 - the model gets the DB error and retries
        return f"Cypher error: {str(e)[:500]}"


TOOLS = [search_mailbox, shipment_file, open_tasks, sop_clause, read_mail, run_cypher]

SYSTEM = f"""You are the handover assistant for Daniel Cho, who on 1 July 2026 takes over the import files of Jiwoo Han
(Senior Import Operator at Meridian Freight Co., Busan). You answer from a knowledge graph built from
Jiwoo's mailbox (292 e-mails, Jan-Jun 2026) and the five company SOPs. Use the tools; never answer from general knowledge.

Working rules:
- Start with search_mailbox or shipment_file; use run_cypher for counts and lists. Read a mail in full when the exact wording matters.
- When a question touches a rule, fetch the clause with sop_clause and say explicitly whether what happened complied with it.
- Answer concisely (under 150 words unless the user asks for a list): facts first, then evidence. Cite every fact with the mail id
  [M-0231] or clause [SOP-03 4.2] it comes from. Dates as YYYY-MM-DD, amounts in USD.
- If the graph does not contain the answer, say so and name what would be needed.

Graph schema for run_cypher:
{GRAPH_SCHEMA}"""

DEMO = [
    "Which carrier declined the free-time extension on the Blue Harbor Foods frozen salmon shipment, and who at the carrier declined it?",
    "Was the free-time extension request on HBL MFCIM260009 compliant with SOP-03 section 4.2? Who paid the demurrage in the end?",
    "What is still open for me (Daniel Cho), and what should I do first next week?",
    "Which SOP rule was broken in the detention case on the Seoul Medica syringes, and what did Minseo Park decide about the invoice?",
]


def ask(client: anthropic.Anthropic, question: str, history: list[str]) -> str:
    content = question
    if history:
        content = "Earlier in this conversation:\n" + "\n".join(history[-6:]) + f"\n\nNew question: {question}"
    runner = client.beta.messages.tool_runner(
        model=schema.LLM_MODEL, max_tokens=4000, system=SYSTEM, tools=TOOLS, max_iterations=12,
        messages=[{"role": "user", "content": content}],
    )
    final = runner.until_done()
    text = "".join(b.text for b in final.content if b.type == "text").strip()
    history.append(f"Q: {question}\nA: {text[:600]}")
    return text


def main() -> None:
    client = anthropic.Anthropic()
    history: list[str] = []
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--demo" in sys.argv:
        for q in DEMO:
            print(f"\n█ {q}")
            print("\n" + ask(client, q, history))
        return
    if args:
        print(ask(client, " ".join(args), history))
        return
    print("Handover agent over data/handover.rl - ask about shipments, decisions, tasks, SOP rules. Ctrl-C to quit.")
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q:
            print("\n" + ask(client, q, history))


if __name__ == "__main__":
    main()
