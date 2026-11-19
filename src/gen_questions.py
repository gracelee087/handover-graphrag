"""Step 5a: 40 evaluation questions with gold answers, derived from world/world.yaml.

Five question types (8 each) chosen to separate retrieval strategies:
  factual     - one fact that sits in one mail or one SOP clause           (vector should cope)
  aggregate   - counts / lists over the whole mailbox                       (needs the graph)
  multihop    - who decided what about which shipment, who owns what        (needs relations)
  compliance  - was X compliant with SOP clause Y, which clause governs     (needs mail<->SOP links)
  handover    - what is open / what to do first                             (needs tasks + status)

Each question carries: gold answer text, `must_contain` keywords for a cheap
check, and the ids of the world objects it rests on.

Run:  uv run python src/gen_questions.py   -> eval/questions.json
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
W = yaml.safe_load((ROOT / "world" / "world.yaml").read_text(encoding="utf-8"))
SH = {s["id"]: s for s in W["shipments"]}
ORG = {o["id"]: o for o in W["organizations"]}
P = {p["id"]: p for p in W["people"]}
INC = {i["id"]: i for i in W["incidents"]}
DEC = {d["id"]: d for d in W["decisions"]}
TASK = {t["id"]: t for t in W["tasks"]}
CL = {c["id"]: c for sop in W["sops"] for sec in sop["sections"] for c in sec["clauses"]}


def expiry(sid: str) -> str:
    s = SH[sid]
    return (date.fromisoformat(s["eta"]) + timedelta(days=s["free_time_days"])).isoformat()


def q(qid: str, qtype: str, question: str, gold: str, must: list[str], basis: list[str]) -> dict:
    return {"id": qid, "type": qtype, "question": question, "gold": gold, "must_contain": must, "basis": basis}


Q: list[dict] = []
s9, s13, s21, s6, s3, s17, s15, s16, s19, s11 = (SH[f"S-{n:03d}"] for n in (9, 13, 21, 6, 3, 17, 15, 16, 19, 11))
by_cons = Counter(s["consignee"] for s in W["shipments"])
by_car = Counter(s["carrier"] for s in W["shipments"])
mails_by_op = sum(1 for th in W["threads"] for b in th["beats"] if b["from"] == "P-001")
open_daniel = [t for t in W["tasks"] if t["owner"] == "P-003" and t["status"] == "open"]
reefers = [s["id"] for s in W["shipments"] if s["container_type"] == "40RF"]
incident_ships = sorted(i["shipment"] for i in W["incidents"])
n_clauses = len(CL)

# ---- factual (8)
Q += [
    q("Q01", "factual", f"What is the container number for HBL {s9['hbl']}?",
      f"Container {s9['container']} ({s9['container_type']}).", [s9["container"]], ["S-009"]),
    q("Q02", "factual", f"On which vessel and voyage was MBL {s13['mbl']} shipped?",
      f"{s13['vessel']} voyage {s13['voyage']}.", [s13["vessel"], s13["voyage"]], ["S-013"]),
    q("Q03", "factual", f"Which customs broker handled the clearance of HBL {s6['hbl']}?",
      f"{ORG[s6['broker']]['name']}.", ["Kim & Park"], ["S-006"]),
    q("Q04", "factual", f"How many days of free time did HBL {s3['hbl']} have and on which date did it expire?",
      f"{s3['free_time_days']} days, expiring {expiry('S-003')}.", [str(s3["free_time_days"]), expiry("S-003")], ["S-003"]),
    q("Q05", "factual", f"What cargo and HS code were declared for HBL {s17['hbl']}?",
      f"{s17['cargo']}, HS {s17['hs_code']}.", [s17["hs_code"]], ["S-017"]),
    q("Q06", "factual", "Who handles the demurrage and detention desk at Hanseatic Marine Lines?",
      f"{P['P-014']['name']} ({P['P-014']['title']}).", ["Marcus Keller"], ["P-014"]),
    q("Q07", "factual", "According to SOP-04, what makes an exception Severity 1?",
      CL["SOP-04-C1.2"]["gist"], ["500", "5 days"], ["SOP-04-C1.2"]),
    q("Q08", "factual", f"What was the demurrage invoice amount on HBL {s9['hbl']} and for how many days?",
      "USD 1,120 for 4 days of demurrage.", ["1,120", "4 days"], ["I-03"]),
]
# ---- aggregate (8)
Q += [
    q("Q09", "aggregate", "How many shipments in the mailbox were consigned to Blue Harbor Foods? List their HBL numbers.",
      f"{by_cons['ORG-BHF']} shipments: " + ", ".join(s["hbl"] for s in W["shipments"] if s["consignee"] == "ORG-BHF") + ".",
      [str(by_cons["ORG-BHF"])], [s["id"] for s in W["shipments"] if s["consignee"] == "ORG-BHF"]),
    q("Q10", "aggregate", "How many e-mails did Jiwoo Han send in total?", f"{mails_by_op} e-mails.", [str(mails_by_op)], ["P-001"]),
    q("Q11", "aggregate", "How many shipments were carried by Pacific Star Shipping?", f"{by_car['ORG-PSS']} shipments.",
      [str(by_car["ORG-PSS"])], [s["id"] for s in W["shipments"] if s["carrier"] == "ORG-PSS"]),
    q("Q12", "aggregate", "Which shipments moved in reefer (40RF) containers?",
      ", ".join(f"{sid} (HBL {SH[sid]['hbl']})" for sid in reefers) + ".", [SH[sid]["hbl"] for sid in reefers], reefers),
    q("Q13", "aggregate", "How many SOP documents and how many clauses in total are in the SOP set?",
      f"{len(W['sops'])} SOP documents with {n_clauses} clauses.", [str(len(W["sops"])), str(n_clauses)], [s["id"] for s in W["sops"]]),
    q("Q14", "aggregate", "Which customs brokerage companies appear in the mailbox, and which consignees does each one serve?",
      "Harbor Gate Customs Brokerage (Daehan Precision, Seoul Medica Supplies) and Kim & Park Customs Services (Blue Harbor Foods).",
      ["Harbor Gate", "Kim & Park"], ["ORG-HGC", "ORG-KPC"]),
    q("Q15", "aggregate", "Which shipments had a problem or exception reported (delay, demurrage, inspection, damage, document issue)? List the shipment ids.",
      ", ".join(incident_ships) + f" ({len(incident_ships)} shipments).", incident_ships[:5], incident_ships),
    q("Q16", "aggregate", "How many open tasks are assigned to Daniel Cho, and what are they?",
      f"{len(open_daniel)} open tasks: " + " | ".join(f"{t['summary']} (due {t['due']})" for t in open_daniel),
      [str(len(open_daniel))], [t["id"] for t in open_daniel]),
]
# ---- multihop (8)
Q += [
    q("Q17", "multihop", f"Who decided how the demurrage on HBL {s9['hbl']} would be split, and what exactly was decided?",
      f"{P[DEC['D-03']['by']]['name']} (Import Team Lead) on {DEC['D-03']['date']}: {DEC['D-03']['summary']}",
      ["Minseo Park", "560", "50"], ["D-03", "I-03"]),
    q("Q18", "multihop", f"What did Grace Yoon decide about the wet-damage claim on HBL {s17['hbl']}, and when?",
      f"On {DEC['D-07']['date']}: {DEC['D-07']['summary']}", ["surveyor", "Orient Container Line"], ["D-07", "I-07"]),
    q("Q19", "multihop", "Which tasks were handed over to Daniel Cho, and which shipment does each one concern?",
      " | ".join(f"{t['summary']} -> {t['shipment'] or 'no shipment'}" for t in open_daniel),
      ["S-017", "S-024", "S-025"], [t["id"] for t in open_daniel]),
    q("Q20", "multihop", "Who is responsible for issuing the debit note to Rapid Haul Logistics for the detention days, and when is it due?",
      f"{P[TASK['T-03']['owner']]['name']} ({P[TASK['T-03']['owner']]['title']}), due {TASK['T-03']['due']}.",
      ["Junho Oh", TASK["T-03"]["due"]], ["T-03", "I-10"]),
    q("Q21", "multihop", f"On HBL {s15['hbl']}, which shipper was asked to arrange the telex release and who was the contact person?",
      f"{ORG['ORG-NMG']['name']}, contact {P['P-027']['name']} ({P['P-027']['title']}).", ["Klaus Richter"], ["I-06", "S-015"]),
    q("Q22", "multihop", "What policy did Grace Yoon decide for Blue Harbor Foods reefer imports, and from when does it apply?",
      f"{DEC['D-11']['summary']} (decided {DEC['D-11']['date']}).", ["Kim & Park", "Q2"], ["D-11"]),
    q("Q23", "multihop", "Which carrier declined the free-time extension on the Blue Harbor Foods frozen salmon shipment, and who at the carrier declined it?",
      f"{ORG[s9['carrier']]['name']}; {P['P-018']['name']} ({P['P-018']['title']}).", ["Pacific Star", "Aiko Mori"], ["I-03", "S-009"]),
    q("Q24", "multihop", f"Who reported the manifest discrepancy on HBL {s21['hbl']}, how big was it, and how was it resolved?",
      f"{P['P-021']['name']} ({ORG['ORG-KPC']['name']}) reported 18 pallets on the manifest vs 20 on the B/L; "
      f"{P['P-002']['name']} decided: {DEC['D-09']['summary']}", ["18", "20", "90"], ["I-09", "D-09"]),
]
# ---- compliance (8)
Q += [
    q("Q25", "compliance", f"Was the free-time extension request on HBL {s9['hbl']} compliant with SOP-03 section 4.2? Explain.",
      f"No. {INC['I-03']['compliance_note']} Clause 4.2: {CL['SOP-03-C4.2']['gist']}", ["No", "2 working days"], ["I-03", "SOP-03-C4.2"]),
    q("Q26", "compliance", f"Which SOP clause requires Severity-1 exceptions to be escalated within 48 hours, and was the detention case on HBL {s16['hbl']} escalated in time?",
      f"SOP-04 section 3.1 ({CL['SOP-04-C3.1']['gist']}). No: {INC['I-10']['compliance_note']}", ["3.1", "48", "No"], ["SOP-04-C3.1", "I-10"]),
    q("Q27", "compliance", f"Did the delivery order request on HBL {s15['hbl']} comply with SOP-02 section 2.4? What happened?",
      f"No. {INC['I-06']['summary']} {INC['I-06']['compliance_note']}", ["No", "telex release"], ["I-06", "SOP-02-C2.4"]),
    q("Q28", "compliance", f"Which SOP clause governs who pays the demurrage on HBL {s9['hbl']}, and how was it applied?",
      f"SOP-03 section 4.3 ({CL['SOP-03-C4.3']['gist']}). Applied in the decision: {DEC['D-03']['summary']}", ["4.3", "560"], ["SOP-03-C4.3", "D-03"]),
    q("Q29", "compliance", f"Was the claim notice for the wet damage on HBL {s17['hbl']} sent within the deadline in SOP-04 section 4.1?",
      f"Yes. {INC['I-07']['compliance_note']} Clause 4.1: {CL['SOP-04-C4.1']['gist']}", ["Yes", "3 days"], ["I-07", "SOP-04-C4.1"]),
    q("Q30", "compliance", f"Which SOP rules did the same-day urgent clearance on HBL {s19['hbl']} breach?",
      f"{INC['I-08']['compliance_note']}", ["5.3", "2.2"], ["I-08", "SOP-01-C5.3", "SOP-05-C2.2"]),
    q("Q31", "compliance", "What reefer free time does SOP-03 section 2.1 give for Pacific Star Shipping, and which open task changes it?",
      f"4 days reefer free time. Task: {TASK['T-07']['summary']} (owner {P[TASK['T-07']['owner']]['name']}, due {TASK['T-07']['due']}).",
      ["4 days", "5 days"], ["SOP-03-C2.1", "T-07"]),
    q("Q32", "compliance", "Which SOP clause requires the trucker's empty-return date to be confirmed in writing, and in which case was that missing?",
      f"SOP-03 section 5.1 ({CL['SOP-03-C5.1']['gist']}). Missing on {s16['id']} (HBL {s16['hbl']}): {INC['I-10']['summary']}",
      ["5.1", s16["hbl"]], ["SOP-03-C5.1", "I-10"]),
]
# ---- handover (8)
s24, s25 = SH["S-024"], SH["S-025"]
Q += [
    q("Q33", "handover", f"What is still outstanding on HBL {s24['hbl']}?",
      f"{TASK['T-04']['summary']} (due {TASK['T-04']['due']}, owner {P['P-003']['name']}).", ["POD", "cost sheet"], ["T-04", "S-024"]),
    q("Q34", "handover", f"What must be done first for HBL {s25['hbl']} and by when?",
      f"{TASK['T-05']['summary']} Due {TASK['T-05']['due']}; ETA {s25['eta']}.", ["Arrival Notice", TASK["T-05"]["due"]], ["T-05", "S-025"]),
    q("Q35", "handover", "Which customer files does Daniel Cho take over from Jiwoo Han, from which date, and who decided it?",
      f"{DEC['D-12']['summary']} Decided by {P[DEC['D-12']['by']]['name']} on {DEC['D-12']['date']}.",
      ["Daehan Precision", "Seoul Medica", "1 July"], ["D-12"]),
    q("Q36", "handover", "What watch-outs did Jiwoo Han list in the handover summary for the successor?",
      "Telex release confirmation before D/O, free-time reminders, and written empty-return dates from truckers.",
      ["telex", "free-time", "empty"], ["TH-handover"]),
    q("Q37", "handover", "Which tasks are still open for the detention case on the Seoul Medica syringes shipment, and who owns each?",
      " | ".join(f"{t['summary']} -> {P[t['owner']]['name']} (due {t['due']})" for t in W["tasks"] if t["incident"] == "I-10" and t["status"] == "open"),
      ["Daniel Cho", "Junho Oh"], ["T-02", "T-03"]),
    q("Q38", "handover", "What does Grace Yoon plan to discuss with Harbor Gate Customs Brokerage in the Q2 vendor review, and why?",
      f"{TASK['T-06']['summary']} Background: {INC['I-05']['summary']} Decision: {DEC['D-05']['summary']}",
      ["non-conformance", "HS"], ["T-06", "I-05", "D-05"]),
    q("Q39", "handover", "Which open tasks are due before 10 July 2026?",
      " | ".join(f"{t['summary']} (due {t['due']}, {P[t['owner']]['name']})" for t in W["tasks"] if t["status"] == "open" and t["due"] < "2026-07-10"),
      [t["due"] for t in W["tasks"] if t["status"] == "open" and t["due"] < "2026-07-10"][:3],
      [t["id"] for t in W["tasks"] if t["status"] == "open" and t["due"] < "2026-07-10"]),
    q("Q40", "handover", "Which shipment is still before arrival at the time of the handover, and what is its ETA?",
      f"{s25['id']} (HBL {s25['hbl']}, {s25['cargo']}), ETA {s25['eta']}.", [s25["hbl"], s25["eta"]], ["S-025"]),
]

assert len(Q) == 40 and len({x["id"] for x in Q}) == 40
out = ROOT / "eval" / "questions.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(Q, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {out}: {Counter(x['type'] for x in Q)}")
