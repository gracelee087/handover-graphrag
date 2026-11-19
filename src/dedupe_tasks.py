"""Step 3c: mark tasks that restate a task from an earlier thread.

Extraction is per thread, so a task handed over in its own thread and then listed again in the
handover summary becomes two Task nodes. Deterministic rule, no LLM:
  same owner + same due date + compatible shipment (equal, or one side unknown)
  + word overlap of the two summaries (Jaccard) >= 0.35
-> the later thread's node gets status 'duplicate' and DUPLICATE_OF -> the original.
Idempotent (resets earlier marks first); run after extract_mail.py.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from store import connect, run  # noqa: E402

STOP = {"the", "a", "an", "to", "of", "for", "on", "in", "and", "with", "is", "be", "by", "at", "from", "as", "that",
        "this", "its", "it", "s", "due", "date"}
MIN_JACCARD = 0.35


def words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP}


def thread_of(task_id: str) -> str:
    return task_id.rsplit("-TASK", 1)[0]


def main() -> None:
    conn = connect()
    run(conn, "MATCH (t:Task {status: 'duplicate'}) SET t.status = 'open'")
    run(conn, "MATCH ()-[r:DUPLICATE_OF]->() DELETE r")
    rows = run(conn, """
        MATCH (t:Task {status: 'open'})-[:ASSIGNED_TO]->(p:Person) WHERE t.due IS NOT NULL
        RETURN t.id AS id, t.due AS due, p.name AS owner, t.shipment_id AS ship, t.summary AS summary ORDER BY t.id""")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["owner"], r["due"])].append(r)
    marked = 0
    for group in groups.values():
        originals: list[dict] = []
        for t in group:                      # ordered by id = by thread order; the first occurrence is the original
            for o in originals:
                if thread_of(o["id"]) == thread_of(t["id"]):
                    continue
                if o["ship"] and t["ship"] and o["ship"] != t["ship"]:
                    continue
                a, b = words(o["summary"]), words(t["summary"])
                j = len(a & b) / len(a | b) if a | b else 0.0
                if j >= MIN_JACCARD:
                    run(conn, "MATCH (d:Task {id: $d}), (o:Task {id: $o}) SET d.status = 'duplicate' MERGE (d)-[:DUPLICATE_OF]->(o)",
                        {"d": t["id"], "o": o["id"]})
                    print(f"{t['id']} duplicates {o['id']} ({t['owner']}, due {t['due']}, jaccard {j:.2f})")
                    marked += 1
                    break
            else:
                originals.append(t)
    print("duplicates marked:", marked)
    print("open tasks now:", run(conn, "MATCH (t:Task {status: 'open'}) RETURN count(t) AS n")[0]["n"])
    conn.close()


if __name__ == "__main__":
    main()
