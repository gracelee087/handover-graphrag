"""Referential-integrity check for world/world.yaml (PLAN step 0 verification).

Run:  uv run python src/check_world.py
Exit code 1 if any problem is found.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
W = yaml.safe_load((ROOT / "world" / "world.yaml").read_text(encoding="utf-8"))

errors: list[str] = []


def err(msg: str) -> None:
    errors.append(msg)


orgs = {o["id"] for o in W["organizations"]}
people = {p["id"] for p in W["people"]}
ships = {s["id"] for s in W["shipments"]}
clauses = {c["id"] for sop in W["sops"] for sec in sop["sections"] for c in sec["clauses"]}
incs = {i["id"] for i in W["incidents"]}
decs = {d["id"] for d in W["decisions"]}
tasks = {t["id"] for t in W["tasks"]}

# uniqueness
for key, coll in [("organizations", orgs), ("people", people), ("shipments", ships),
                  ("incidents", incs), ("decisions", decs), ("tasks", tasks)]:
    if len(coll) != len(W[key]):
        err(f"duplicate ids in {key}")
if len(clauses) != sum(len(sec["clauses"]) for sop in W["sops"] for sec in sop["sections"]):
    err("duplicate clause ids")

emails = [p["email"] for p in W["people"]]
if len(set(emails)) != len(emails):
    err("duplicate e-mail addresses")

for p in W["people"]:
    if p["org"] not in orgs:
        err(f"{p['id']} org {p['org']} unknown")

for s in W["shipments"]:
    for f in ("carrier", "broker", "consignee"):
        if s[f] not in orgs:
            err(f"{s['id']} {f} {s[f]} unknown")
    if s.get("shipper") and s["shipper"] not in orgs:
        err(f"{s['id']} shipper unknown")
    if s["has_incident"] and s.get("incident") not in incs:
        err(f"{s['id']} flagged incident but no incident id")

for i in W["incidents"]:
    if i["shipment"] not in ships:
        err(f"{i['id']} shipment unknown")
    for c in i["clauses"]:
        if c not in clauses:
            err(f"{i['id']} clause {c} unknown")
    if i["severity"] == 1 and i["compliant"] and not i["escalated"]:
        err(f"{i['id']} severity 1 marked compliant but never escalated")

for dd in W["decisions"]:
    if dd["by"] not in people:
        err(f"{dd['id']} decided by unknown person")
    if dd["incident"] and dd["incident"] not in incs:
        err(f"{dd['id']} incident unknown")
    if dd["shipment"] and dd["shipment"] not in ships:
        err(f"{dd['id']} shipment unknown")

for t in W["tasks"]:
    if t["owner"] not in people:
        err(f"{t['id']} owner unknown")
    if t["shipment"] and t["shipment"] not in ships:
        err(f"{t['id']} shipment unknown")
    if t["incident"] and t["incident"] not in incs:
        err(f"{t['id']} incident unknown")

# threads / beats
start, end = date.fromisoformat(W["meta"]["period_start"]), date.fromisoformat(W["meta"]["period_end"])
mail_ids: list[str] = []
mentioned_dec, mentioned_task, mentioned_inc_ship = set(), set(), set()
for th in W["threads"]:
    if th["shipment"] and th["shipment"] not in ships:
        err(f"{th['id']} shipment unknown")
    if th["incident"] and th["incident"] not in incs:
        err(f"{th['id']} incident unknown")
    if not th["beats"]:
        err(f"{th['id']} has no beats")
    prev = None
    for b in th["beats"]:
        mail_ids.append(b["id"])
        if b["from"] not in people:
            err(f"{b['id']} from unknown {b['from']}")
        for r in b["to"] + b["cc"]:
            if r not in people:
                err(f"{b['id']} recipient unknown {r}")
        if b["from"] in b["to"]:
            err(f"{b['id']} sender is also recipient")
        dt = date.fromisoformat(b["date"])
        if not (start <= dt <= end):
            err(f"{b['id']} date {b['date']} outside period")
        if prev and dt < prev:
            err(f"{b['id']} date goes backwards inside {th['id']}")
        prev = dt
        for d_id in b.get("decisions", []):
            if d_id not in decs:
                err(f"{b['id']} decision {d_id} unknown")
            mentioned_dec.add(d_id)
        for t_id in b.get("tasks", []):
            if t_id not in tasks:
                err(f"{b['id']} task {t_id} unknown")
            mentioned_task.add(t_id)
        if th["incident"]:
            mentioned_inc_ship.add(th["incident"])
if len(set(mail_ids)) != len(mail_ids):
    err("duplicate mail ids")

for d_id in decs - mentioned_dec:
    err(f"decision {d_id} never appears in a mail")
for i_id in incs - mentioned_inc_ship:
    err(f"incident {i_id} has no thread")
open_tasks = {t["id"] for t in W["tasks"] if t["status"] == "open"}
for t_id in open_tasks - mentioned_task:
    err(f"open task {t_id} never appears in a mail")
for s_id in ships - {th["shipment"] for th in W["threads"] if th["shipment"]}:
    err(f"shipment {s_id} has no thread")

n_mail = len(mail_ids)
print(f"mails={n_mail} threads={len(W['threads'])} clauses={len(clauses)} "
      f"non_compliant={sum(1 for i in W['incidents'] if not i['compliant'])}/{len(incs)}")
if not 150 <= n_mail <= 300:
    err(f"mail count {n_mail} outside 150-300 target")

if errors:
    print("\n".join(f"ERROR {e}" for e in errors))
    sys.exit(1)
print("OK - world.yaml is consistent")
