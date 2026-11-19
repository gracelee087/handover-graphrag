"""Generate world/world.yaml - the ground truth for the whole project.

Everything factual (people, organizations, shipments, incidents, decisions,
tasks, SOP clause structure, and the beat-by-beat script of every mail thread)
is decided HERE by code with a fixed seed. The LLM later only turns the beats
into English prose (gen_mail.py) and clause gists into clause text (gen_sop.py).

All incident / decision / task dates are derived from each shipment's ETA so the
timeline stays consistent if ETAs change.

Run:  uv run python src/gen_world.py
"""
from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import yaml

SEED = 20260119
rng = random.Random(SEED)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "world" / "world.yaml"

COMPANY = "Meridian Freight Co."
DOMAIN = "meridianfreight.example"
PERIOD_START = date(2026, 1, 5)
PERIOD_END = date(2026, 6, 30)
FIRST_ETA = date(2026, 1, 16)
ETA_STEP_DAYS = 6
LAST_ETA = date(2026, 6, 29)      # S-025, still pre-arrival at handover


def plus(iso: str, days: int) -> str:
    return (date.fromisoformat(iso) + timedelta(days=days)).isoformat()


def workday(iso: str) -> str:
    """Shift weekend dates to the following Monday."""
    dt = date.fromisoformat(iso)
    while dt.weekday() >= 5:
        dt += timedelta(days=1)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# Organizations
# ---------------------------------------------------------------------------
ORGS = [
    {"id": "ORG-MFC", "name": COMPANY, "type": "forwarder", "domain": DOMAIN},
    {"id": "ORG-HML", "name": "Hanseatic Marine Lines", "type": "carrier", "domain": "hml-lines.example", "mbl_prefix": "HMLU"},
    {"id": "ORG-OCL", "name": "Orient Container Line", "type": "carrier", "domain": "ocl-shipping.example", "mbl_prefix": "OCLU"},
    {"id": "ORG-PSS", "name": "Pacific Star Shipping", "type": "carrier", "domain": "pacstar.example", "mbl_prefix": "PSSU"},
    {"id": "ORG-HGC", "name": "Harbor Gate Customs Brokerage", "type": "broker", "domain": "harborgate.example"},
    {"id": "ORG-KPC", "name": "Kim & Park Customs Services", "type": "broker", "domain": "kimpark-customs.example"},
    {"id": "ORG-DPC", "name": "Daehan Precision Co.", "type": "consignee", "domain": "daehanprecision.example"},
    {"id": "ORG-SMS", "name": "Seoul Medica Supplies", "type": "consignee", "domain": "seoulmedica.example"},
    {"id": "ORG-BHF", "name": "Blue Harbor Foods", "type": "consignee", "domain": "blueharborfoods.example"},
    {"id": "ORG-NMG", "name": "Nordlicht Maschinenbau GmbH", "type": "shipper", "domain": "nordlicht-mb.example"},
    {"id": "ORG-RHL", "name": "Rapid Haul Logistics", "type": "trucker", "domain": "rapidhaul.example"},
    {"id": "ORG-BNT", "name": "Busan New Port Terminal", "type": "terminal", "domain": "bnpt.example"},
]
ORG = {o["id"]: o for o in ORGS}


# ---------------------------------------------------------------------------
# People (12 internal + 18 external)
# ---------------------------------------------------------------------------
def person(pid, name, org, title, internal=False):
    first, last = name.split(" ", 1)
    local = f"{first.lower()}.{last.lower().replace(' ', '')}"
    return {"id": pid, "name": name, "org": org, "title": title,
            "email": f"{local}@{ORG[org]['domain']}", "is_internal": internal}


PEOPLE = [
    person("P-001", "Jiwoo Han", "ORG-MFC", "Senior Import Operator (leaving 30 June 2026)", True),
    person("P-002", "Minseo Park", "ORG-MFC", "Import Team Lead", True),
    person("P-003", "Daniel Cho", "ORG-MFC", "Import Operator (successor)", True),
    person("P-004", "Yuna Seo", "ORG-MFC", "Export Operator", True),
    person("P-005", "Thomas Kwon", "ORG-MFC", "Sales Manager", True),
    person("P-006", "Hyejin Lim", "ORG-MFC", "Sales Representative", True),
    person("P-007", "Sora Baek", "ORG-MFC", "Accounting Manager", True),
    person("P-008", "Junho Oh", "ORG-MFC", "Accounts Payable Clerk", True),
    person("P-009", "Grace Yoon", "ORG-MFC", "Operations Manager", True),
    person("P-010", "Eunji Kang", "ORG-MFC", "Customer Service", True),
    person("P-011", "Hoon Jang", "ORG-MFC", "Documentation Clerk", True),
    person("P-012", "Sejin Ryu", "ORG-MFC", "IT & Admin", True),
    person("P-013", "Lena Vogt", "ORG-HML", "Import Customer Service"),
    person("P-014", "Marcus Keller", "ORG-HML", "Demurrage & Detention Desk"),
    person("P-015", "Wei Zhang", "ORG-OCL", "Import Documentation"),
    person("P-016", "Mei Lin", "ORG-OCL", "Key Account Manager"),
    person("P-017", "Ryan Tanaka", "ORG-PSS", "Import Customer Service"),
    person("P-018", "Aiko Mori", "ORG-PSS", "Equipment Control"),
    person("P-019", "Sunwoo Choi", "ORG-HGC", "Licensed Customs Broker"),
    person("P-020", "Nara Shin", "ORG-HGC", "Clearance Coordinator"),
    person("P-021", "Dongwook Kim", "ORG-KPC", "Licensed Customs Broker"),
    person("P-022", "Haneul Park", "ORG-KPC", "Clearance Assistant"),
    person("P-023", "Jaehyun Lee", "ORG-DPC", "Purchasing Manager"),
    person("P-024", "Soyeon Moon", "ORG-DPC", "Logistics Coordinator"),
    person("P-025", "Insoo Hwang", "ORG-SMS", "Supply Chain Manager"),
    person("P-026", "Bora Jung", "ORG-BHF", "Import Buyer"),
    person("P-027", "Klaus Richter", "ORG-NMG", "Export Sales"),
    person("P-028", "Kyungho Nam", "ORG-RHL", "Dispatch Manager"),
    person("P-029", "Woojin Seok", "ORG-RHL", "Dispatcher"),
    person("P-030", "Taeyang Koo", "ORG-BNT", "Gate Operations"),
]
P = {p["id"]: p for p in PEOPLE}

CARRIER_CS = {"ORG-HML": "P-013", "ORG-OCL": "P-015", "ORG-PSS": "P-017"}
CARRIER_DND = {"ORG-HML": "P-014", "ORG-OCL": "P-016", "ORG-PSS": "P-018"}
BROKER_MAIN = {"ORG-HGC": "P-019", "ORG-KPC": "P-021"}
BROKER_ASSIST = {"ORG-HGC": "P-020", "ORG-KPC": "P-022"}
CONSIGNEE_CONTACT = {"ORG-DPC": "P-024", "ORG-SMS": "P-025", "ORG-BHF": "P-026"}
CONSIGNEE_MGR = {"ORG-DPC": "P-023", "ORG-SMS": "P-025", "ORG-BHF": "P-026"}

OP = "P-001"      # predecessor runs every file until handover
LEAD = "P-002"
JUNIOR = "P-003"
DOCS = "P-011"
ACCT = "P-008"
OPS_MGR = "P-009"
SALES = "P-005"
TRUCK_MGR, TRUCK_DISP = "P-028", "P-029"
SHIPPER_CT = "P-027"

# ---------------------------------------------------------------------------
# Shipments (25)
# ---------------------------------------------------------------------------
VESSELS = {
    "ORG-HML": [("HML Bremen", "026W"), ("HML Lubeck", "031W"), ("HML Kiel", "035W")],
    "ORG-OCL": [("OCL Ningbo Star", "2602E"), ("OCL Qingdao Star", "2609E"), ("OCL Dalian Star", "2615E")],
    "ORG-PSS": [("Pacific Aurora", "014N"), ("Pacific Meridian", "019N"), ("Pacific Borealis", "023N")],
}
ROUTES = {
    "ORG-HML": [("DEHAM", "Hamburg"), ("NLRTM", "Rotterdam"), ("BEANR", "Antwerp")],
    "ORG-OCL": [("CNSHA", "Shanghai"), ("CNNGB", "Ningbo"), ("CNTAO", "Qingdao")],
    "ORG-PSS": [("USLAX", "Los Angeles"), ("TWKHH", "Kaohsiung"), ("JPYOK", "Yokohama")],
}
CARGO = {
    "ORG-DPC": [("CNC machine spare parts", "8466.93"), ("precision bearings", "8482.10"),
                ("hydraulic press components", "8462.29"), ("industrial sensors", "9031.80")],
    "ORG-SMS": [("disposable syringes", "9018.31"), ("diagnostic reagent kits", "3822.00"),
                ("surgical gloves", "4015.12"), ("infusion pumps", "9018.90")],
    "ORG-BHF": [("frozen shrimp", "0306.17"), ("canned tuna", "1604.14"),
                ("olive oil", "1509.10"), ("frozen salmon fillets", "0304.81")],
}
# consignee per shipment: incident shipments are pinned so stories stay consistent
CONSIGNEE_OVERRIDE = {3: "ORG-DPC", 6: "ORG-BHF", 9: "ORG-BHF", 11: "ORG-SMS", 13: "ORG-DPC",
                      15: "ORG-DPC", 16: "ORG-SMS", 17: "ORG-SMS", 19: "ORG-DPC", 21: "ORG-BHF"}
CARGO_OVERRIDE = {6: 0, 9: 3, 11: 3, 13: 1, 17: 1, 19: 3}   # index into CARGO[consignee]

SHIPMENTS = []
_consignees = ["ORG-DPC", "ORG-SMS", "ORG-BHF"]
_carriers = ["ORG-HML", "ORG-OCL", "ORG-PSS"]
for i in range(1, 26):
    sid = f"S-{i:03d}"
    carrier = _carriers[(i - 1) % 3]
    consignee = CONSIGNEE_OVERRIDE.get(i, _consignees[(i * 2 + i // 3) % 3])
    broker = "ORG-KPC" if consignee == "ORG-BHF" else "ORG-HGC"
    vessel, voyage = VESSELS[carrier][(i // 3) % 3]
    pol_code, pol_name = ROUTES[carrier][i % 3]
    eta = LAST_ETA if i == 25 else FIRST_ETA + timedelta(days=ETA_STEP_DAYS * (i - 1) + rng.randint(0, 1))
    cargo, hs = CARGO[consignee][CARGO_OVERRIDE.get(i, i % 4)]
    shipper = "ORG-NMG" if consignee == "ORG-DPC" and (carrier == "ORG-HML" or i == 15) else None
    reefer = consignee == "ORG-BHF" and cargo.startswith("frozen")
    SHIPMENTS.append({
        "id": sid,
        "mbl": f"{ORG[carrier]['mbl_prefix']}{260000 + i * 37}",
        "hbl": f"MFCIM26{i:04d}",
        "container": f"{ORG[carrier]['mbl_prefix'][:4]}{rng.randint(1000000, 9999999)}",
        "container_type": "40RF" if reefer else rng.choice(["40HC", "20GP"]),
        "vessel": vessel, "voyage": voyage,
        "pol": pol_code, "pol_name": pol_name, "pod": "KRPUS", "pod_name": "Busan",
        "eta": eta.isoformat(), "ata": None if i == 25 else eta.isoformat(),
        "free_time_days": 10 if consignee == "ORG-BHF" else 7,   # SOP-03 2.1: BHF contract +3
        "carrier": carrier, "broker": broker, "consignee": consignee, "shipper": shipper,
        "cargo": cargo, "hs_code": hs,
        "cargo_value_usd": rng.randint(18000, 240000),
        "status": "pre_arrival" if i == 25 else ("awaiting_pod" if i == 24 else "delivered"),
        "has_incident": False, "incident": None,
    })
SH = {s["id"]: s for s in SHIPMENTS}

# ---------------------------------------------------------------------------
# SOP documents: structure + clause gists (LLM writes prose later)
# ---------------------------------------------------------------------------
SOPS = [
    {"id": "SOP-01", "title": "Import Clearance Standard Operating Procedure", "version": "v3.2",
     "effective_date": "2025-09-01", "filename": "SOP-01_Import_Clearance.pdf",
     "sections": [
         {"number": "1", "title": "Purpose and Scope", "clauses": [
             {"number": "1.1", "title": "Purpose", "gist": "Defines the mandatory steps for sea import clearance handled by the Import Operations team."},
             {"number": "1.2", "title": "Scope", "gist": "Applies to all FCL and LCL sea imports where Meridian acts as forwarder or NVOCC; air imports are covered by SOP-06."}]},
         {"number": "2", "title": "Roles and Responsibilities", "clauses": [
             {"number": "2.1", "title": "Import Operator", "gist": "Owns the file from pre-alert to delivery; single point of contact for consignee, broker, carrier, trucker."},
             {"number": "2.2", "title": "Import Team Lead", "gist": "Approves exceptions, signs off on cost absorption up to USD 2,000, receives escalations."},
             {"number": "2.3", "title": "Documentation Clerk", "gist": "Issues House B/L, maintains document register, archives originals."}]},
         {"number": "3", "title": "Pre-arrival", "clauses": [
             {"number": "3.1", "title": "Pre-alert review", "gist": "Operator reviews the pre-alert within 1 working day of receipt and confirms HBL/MBL data match."},
             {"number": "3.2", "title": "Arrival Notice", "gist": "Arrival Notice is sent to consignee no later than 3 working days before ETA; it must quote MBL, HBL, container, vessel/voyage, ETA and free time."},
             {"number": "3.3", "title": "Document collection", "gist": "Commercial invoice, packing list and C/O (if preferential) are requested from the consignee at the time of the Arrival Notice; see SOP-02 for the checklist."}]},
         {"number": "4", "title": "Customs Declaration", "clauses": [
             {"number": "4.1", "title": "Broker instruction", "gist": "A complete document set is forwarded to the appointed customs broker at least 2 working days before ETA."},
             {"number": "4.2", "title": "HS code verification", "gist": "Operator cross-checks the broker's proposed HS code against the consignee's import history; any change is confirmed in writing with the consignee before filing."},
             {"number": "4.3", "title": "Duty and tax payment", "gist": "Duty is paid by the consignee directly unless a disbursement is pre-approved by Accounting."}]},
         {"number": "5", "title": "Inspection and Release", "clauses": [
             {"number": "5.1", "title": "Inspection notice", "gist": "If customs selects the shipment for physical inspection, the operator informs the consignee within 1 working day and records the inspection in the file."},
             {"number": "5.2", "title": "Delivery Order", "gist": "D/O is requested from the carrier only after customs release AND confirmed O/BL surrender or telex release."},
             {"number": "5.3", "title": "Trucking booking", "gist": "Trucking is booked at least 1 working day before planned pick-up and the consignee is given the delivery slot in writing."}]},
         {"number": "6", "title": "File Closure", "clauses": [
             {"number": "6.1", "title": "Proof of delivery", "gist": "Signed POD is obtained within 3 working days of delivery and attached to the file."},
             {"number": "6.2", "title": "Invoicing handover", "gist": "Cost sheet handed to Accounting within 5 working days of delivery."}]},
     ]},
    {"id": "SOP-02", "title": "Import Document Checklist and B/L Handling", "version": "v2.0",
     "effective_date": "2025-06-15", "filename": "SOP-02_Document_Checklist_BL_Handling.pdf",
     "sections": [
         {"number": "1", "title": "Mandatory Document Set", "clauses": [
             {"number": "1.1", "title": "Checklist", "gist": "Table of required documents per cargo type: commercial invoice, packing list, B/L, C/O, import licence (medical devices), health certificate (food)."},
             {"number": "1.2", "title": "Completeness check", "gist": "Operator ticks the checklist in the file before instructing the broker; missing items are chased with the consignee in writing."}]},
         {"number": "2", "title": "Bill of Lading", "clauses": [
             {"number": "2.1", "title": "Master vs House", "gist": "MBL is issued by the carrier to Meridian; HBL is issued by Meridian to the consignee. Both numbers appear on every mail subject about the shipment."},
             {"number": "2.2", "title": "Original B/L", "gist": "Where an original B/L is issued, all three originals or one duly endorsed original must be surrendered before cargo release."},
             {"number": "2.3", "title": "Telex release / Sea waybill", "gist": "Telex release confirmation from the carrier replaces O/BL surrender; a screenshot of the confirmation is filed."},
             {"number": "2.4", "title": "Release confirmation before D/O", "gist": "No D/O request is sent to the carrier until O/BL surrender or telex release is confirmed and recorded in the file."}]},
         {"number": "3", "title": "Amendments", "clauses": [
             {"number": "3.1", "title": "B/L correction", "gist": "Any B/L amendment after vessel departure requires the shipper's written request and consignee's acknowledgement; amendment fees are quoted to the requesting party."},
             {"number": "3.2", "title": "Manifest correction", "gist": "Manifest corrections at destination must be filed before the vessel's ATA; late corrections are escalated to the Team Lead."}]},
         {"number": "4", "title": "Archiving", "clauses": [
             {"number": "4.1", "title": "Retention", "gist": "All import documents are retained for 5 years in the document management system."}]},
     ]},
    {"id": "SOP-03", "title": "Demurrage and Detention Avoidance and Free-time Policy", "version": "v4.1",
     "effective_date": "2025-11-01", "filename": "SOP-03_Demurrage_Detention_Freetime.pdf",
     "sections": [
         {"number": "1", "title": "Definitions", "clauses": [
             {"number": "1.1", "title": "Free time", "gist": "The number of calendar days from container discharge during which no demurrage is charged by the carrier."},
             {"number": "1.2", "title": "Demurrage vs Detention", "gist": "Demurrage accrues while the container is inside the terminal; detention accrues after gate-out until empty return."}]},
         {"number": "2", "title": "Standard Free Time", "clauses": [
             {"number": "2.1", "title": "Contract free time table", "gist": "Table: Hanseatic Marine Lines 7 days dry / 5 days reefer; Orient Container Line 7 days dry / 5 days reefer; Pacific Star Shipping 7 days dry / 4 days reefer; Blue Harbor Foods contract adds 3 days on all carriers."},
             {"number": "2.2", "title": "Recording", "gist": "Free-time expiry date is recorded in the file on the day the Arrival Notice is issued."}]},
         {"number": "3", "title": "Monitoring", "clauses": [
             {"number": "3.1", "title": "Daily watch list", "gist": "Operator reviews all containers with free time expiring within 3 days every morning."},
             {"number": "3.2", "title": "Consignee reminder", "gist": "Consignee is reminded in writing 3 working days before free-time expiry if customs release or pick-up is not yet confirmed."}]},
         {"number": "4", "title": "Extension and Charges", "clauses": [
             {"number": "4.1", "title": "Extension request", "gist": "Free-time extension is requested from the carrier's D&D desk in writing when a delay is foreseeable."},
             {"number": "4.2", "title": "Extension timing", "gist": "The extension request must be sent no later than 2 working days before free time expires; requests after expiry are generally refused by carriers."},
             {"number": "4.3", "title": "Charge allocation", "gist": "Demurrage caused by consignee document delay is billed to the consignee; demurrage caused by Meridian's error is absorbed with Team Lead approval and recorded as a cost-absorption decision."},
             {"number": "4.4", "title": "Dispute", "gist": "D&D invoices are disputed with the carrier within 7 days of receipt when the charge contradicts the recorded free time."}]},
         {"number": "5", "title": "Empty Return", "clauses": [
             {"number": "5.1", "title": "Return deadline", "gist": "Trucker confirms empty return date in writing; operator records the return date to close detention exposure."}]},
     ]},
    {"id": "SOP-04", "title": "Exception and Escalation Policy", "version": "v2.3",
     "effective_date": "2025-10-01", "filename": "SOP-04_Exception_Escalation.pdf",
     "sections": [
         {"number": "1", "title": "Exception Categories", "clauses": [
             {"number": "1.1", "title": "Categories", "gist": "Table of exception categories: vessel delay > 3 days, customs inspection, document discrepancy, demurrage/detention exposure, cargo damage or shortage, regulatory hold."},
             {"number": "1.2", "title": "Severity", "gist": "Severity 1: potential cost over USD 500 or delivery delay over 5 days; Severity 2: all other exceptions."}]},
         {"number": "2", "title": "Reporting", "clauses": [
             {"number": "2.1", "title": "Consignee notification", "gist": "Consignee is informed of any exception within 1 working day of Meridian becoming aware of it."},
             {"number": "2.2", "title": "Incident record", "gist": "Each exception is logged in the file with date detected, cause, expected impact and owner."}]},
         {"number": "3", "title": "Escalation", "clauses": [
             {"number": "3.1", "title": "48-hour escalation", "gist": "Severity 1 exceptions are escalated to the Import Team Lead in writing within 48 hours of detection."},
             {"number": "3.2", "title": "Management escalation", "gist": "Exceptions with potential cost over USD 2,000 or a claim against Meridian are escalated to the Operations Manager by the Team Lead."},
             {"number": "3.3", "title": "Decision recording", "gist": "Every escalation ends with a recorded decision stating what was decided, by whom and on which date."}]},
         {"number": "4", "title": "Claims", "clauses": [
             {"number": "4.1", "title": "Claim notice", "gist": "Cargo damage or shortage noted at delivery is notified to the carrier in writing within 3 days of delivery to preserve recourse."},
             {"number": "4.2", "title": "Evidence", "gist": "Photos, delivery receipt remarks and a survey report (if value over USD 10,000) are collected before any settlement discussion."}]},
     ]},
    {"id": "SOP-05", "title": "Customs Broker and Vendor Management Policy", "version": "v1.4",
     "effective_date": "2025-04-01", "filename": "SOP-05_Broker_Vendor_Management.pdf",
     "sections": [
         {"number": "1", "title": "Approved Vendors", "clauses": [
             {"number": "1.1", "title": "Approved list", "gist": "Only brokers and truckers on the approved vendor list may be instructed; the list is maintained by the Operations Manager."},
             {"number": "1.2", "title": "Broker allocation", "gist": "Table: Harbor Gate Customs Brokerage handles Daehan Precision and Seoul Medica; Kim & Park Customs Services handles Blue Harbor Foods (food and agricultural)."}]},
         {"number": "2", "title": "Instruction and Quotation", "clauses": [
             {"number": "2.1", "title": "Written instruction", "gist": "Brokers and truckers are instructed by e-mail with the HBL number in the subject; verbal instructions are confirmed in writing the same day."},
             {"number": "2.2", "title": "Rate validity", "gist": "Vendor rates are agreed annually; any charge outside the rate sheet requires a quotation approved by the Team Lead before the service is performed."}]},
         {"number": "3", "title": "Performance Review", "clauses": [
             {"number": "3.1", "title": "Quarterly review", "gist": "Operations Manager reviews broker clearance lead time and trucker on-time rate each quarter."},
             {"number": "3.2", "title": "Non-conformance", "gist": "Repeated errors by a vendor are recorded and discussed with the vendor; two recorded non-conformances in a quarter trigger a review of the allocation."}]},
         {"number": "4", "title": "Invoices", "clauses": [
             {"number": "4.1", "title": "Invoice check", "gist": "Vendor invoices are checked against the instruction and the rate sheet by the operator before Accounting pays them."}]},
     ]},
]
for sop in SOPS:
    for sec in sop["sections"]:
        sec["id"] = f"{sop['id']}-S{sec['number']}"
        for cl in sec["clauses"]:
            cl["id"] = f"{sop['id']}-C{cl['number']}"


# ---------------------------------------------------------------------------
# Incidents (10) - dates relative to the shipment ETA; verdict against clauses
# ---------------------------------------------------------------------------
def E(sid: str, days: int) -> str:
    return workday(plus(SH[sid]["eta"], days))


def expiry(sid: str) -> str:
    s = SH[sid]
    return plus(s["eta"], s["free_time_days"])


INCIDENTS = [
    {"id": "I-01", "shipment": "S-003", "kind": "vessel_delay", "severity": 1,
     "detected": E("S-003", -6), "escalated": E("S-003", -6), "consignee_informed": E("S-003", -6),
     "summary": "Vessel delayed 6 days by Shanghai port congestion; consignee production line at risk.",
     "clauses": ["SOP-04-C2.1", "SOP-04-C3.1"], "compliant": True,
     "compliance_note": "Consignee informed same day; Team Lead escalated the same day (within 48h)."},
    {"id": "I-02", "shipment": "S-006", "kind": "missing_documents", "severity": 2,
     "detected": E("S-006", -3), "escalated": None, "consignee_informed": E("S-006", -3),
     "summary": "Health certificate missing for frozen shrimp; broker could not file; free time at risk.",
     "clauses": ["SOP-02-C1.2", "SOP-03-C3.2"], "compliant": True,
     "compliance_note": "Missing item chased in writing the same day; written reminder sent 3 working days before free-time expiry."},
    {"id": "I-03", "shipment": "S-009", "kind": "demurrage", "severity": 1,
     "detected": workday(plus(expiry("S-009"), 1)), "escalated": workday(plus(expiry("S-009"), 6)),
     "consignee_informed": workday(plus(expiry("S-009"), 1)),
     "summary": "Consignee asked to hold pick-up; extension requested only on the day free time expired; 4 days demurrage USD 1,120.",
     "clauses": ["SOP-03-C4.2", "SOP-04-C3.1"], "compliant": False,
     "compliance_note": "Extension request sent on expiry day (SOP-03 4.2 requires 2 working days before). Escalation took 5 days (SOP-04 3.1 requires 48h)."},
    {"id": "I-04", "shipment": "S-011", "kind": "customs_inspection", "severity": 2,
     "detected": E("S-011", 1), "escalated": None, "consignee_informed": E("S-011", 1),
     "summary": "Physical inspection ordered on infusion pumps; import licence number queried.",
     "clauses": ["SOP-01-C5.1", "SOP-04-C2.1"], "compliant": True,
     "compliance_note": "Consignee informed within one working day; inspection recorded."},
    {"id": "I-05", "shipment": "S-013", "kind": "hs_code_dispute", "severity": 2,
     "detected": E("S-013", 3), "escalated": None, "consignee_informed": None,
     "summary": "Broker filed under HS 8482.99 instead of the consignee's usual 8482.10 without written confirmation; duty difference USD 380.",
     "clauses": ["SOP-01-C4.2"], "compliant": False,
     "compliance_note": "HS code change filed without written confirmation from consignee."},
    {"id": "I-06", "shipment": "S-015", "kind": "do_release_hold", "severity": 1,
     "detected": E("S-015", 0), "escalated": E("S-015", 1), "consignee_informed": E("S-015", 0),
     "summary": "D/O requested before telex release was confirmed; carrier refused release, 2 days lost, demurrage USD 560.",
     "clauses": ["SOP-02-C2.4", "SOP-04-C3.1"], "compliant": False,
     "compliance_note": "D/O requested before release confirmation (SOP-02 2.4). Escalation itself was on time."},
    {"id": "I-07", "shipment": "S-017", "kind": "cargo_damage_claim", "severity": 1,
     "detected": E("S-017", 4), "escalated": E("S-017", 4), "consignee_informed": E("S-017", 4),
     "summary": "Wet damage on 14 cartons of diagnostic reagent kits at delivery; claim notice to carrier within 3 days.",
     "clauses": ["SOP-04-C4.1", "SOP-04-C4.2", "SOP-04-C3.2"], "compliant": True,
     "compliance_note": "Claim notice sent the next day; photos and survey arranged; escalated to Ops Manager."},
    {"id": "I-08", "shipment": "S-019", "kind": "urgent_clearance", "severity": 2,
     "detected": E("S-019", 1), "escalated": None, "consignee_informed": E("S-019", 1),
     "summary": "Consignee requested same-day clearance for production stop; broker expedited; trucking booked same day.",
     "clauses": ["SOP-01-C5.3", "SOP-05-C2.2"], "compliant": False,
     "compliance_note": "Trucking booked same day (SOP-01 5.3 requires 1 working day notice) and expedite fee incurred without approved quotation (SOP-05 2.2)."},
    {"id": "I-09", "shipment": "S-021", "kind": "manifest_discrepancy", "severity": 2,
     "detected": E("S-021", 0), "escalated": E("S-021", 1), "consignee_informed": E("S-021", 0),
     "summary": "Carrier manifest showed 18 pallets vs B/L 20; correction filed after ATA, escalated to Team Lead.",
     "clauses": ["SOP-02-C3.2", "SOP-04-C2.1"], "compliant": True,
     "compliance_note": "Late correction correctly escalated to Team Lead as SOP-02 3.2 requires."},
    {"id": "I-10", "shipment": "S-016", "kind": "detention", "severity": 1,
     "detected": E("S-016", 21), "escalated": None, "consignee_informed": E("S-016", 23),
     "summary": "Empty container returned 6 days late by trucker; detention invoice USD 900; no written return date recorded.",
     "clauses": ["SOP-03-C5.1", "SOP-04-C3.1", "SOP-04-C2.1"], "compliant": False,
     "compliance_note": "No written empty-return date (SOP-03 5.1); never escalated within 48h (SOP-04 3.1); consignee informed after 2 days (SOP-04 2.1)."},
]
for inc in INCIDENTS:
    SH[inc["shipment"]]["has_incident"] = True
    SH[inc["shipment"]]["incident"] = inc["id"]
INC = {i["id"]: i for i in INCIDENTS}

# ---------------------------------------------------------------------------
# Decisions (12) and Tasks (8 open + 2 done)
# ---------------------------------------------------------------------------
DECISIONS = [
    {"id": "D-01", "date": E("S-003", -5), "by": LEAD, "incident": "I-01", "shipment": "S-003",
     "summary": "Accept the delay; no re-routing. Daily ETA update to Daehan Precision until berthing."},
    {"id": "D-02", "date": workday(plus(expiry("S-006"), -2)), "by": OP, "incident": "I-02", "shipment": "S-006",
     "summary": "File the declaration on the day the health certificate arrives; use Kim & Park's priority slot."},
    {"id": "D-03", "date": workday(plus(INC["I-03"]["escalated"], 1)), "by": LEAD, "incident": "I-03", "shipment": "S-009",
     "summary": "Meridian absorbs 50% of USD 1,120 demurrage (USD 560) because the extension request was late; the rest billed to Blue Harbor Foods."},
    {"id": "D-04", "date": E("S-011", 3), "by": OP, "incident": "I-04", "shipment": "S-011",
     "summary": "Attend the inspection with the broker; consignee sends the import licence copy directly to customs."},
    {"id": "D-05", "date": E("S-013", 8), "by": LEAD, "incident": "I-05", "shipment": "S-013",
     "summary": "Keep the filed HS code (broker's classification confirmed correct); record a non-conformance against Harbor Gate for filing without written confirmation."},
    {"id": "D-06", "date": workday(plus(INC["I-06"]["escalated"], 1)), "by": LEAD, "incident": "I-06", "shipment": "S-015",
     "summary": "Absorb USD 560 demurrage as Meridian error; re-train the team on SOP-02 2.4 release confirmation."},
    {"id": "D-07", "date": E("S-017", 8), "by": OPS_MGR, "incident": "I-07", "shipment": "S-017",
     "summary": "Appoint an independent surveyor; pursue the claim against Orient Container Line; Seoul Medica is kept informed weekly."},
    {"id": "D-08", "date": E("S-019", 1), "by": OP, "incident": "I-08", "shipment": "S-019",
     "summary": "Approve broker expedite fee USD 150 verbally and book Rapid Haul for same-day pick-up to prevent the Daehan line stop."},
    {"id": "D-09", "date": E("S-021", 2), "by": LEAD, "incident": "I-09", "shipment": "S-021",
     "summary": "File the manifest correction at Meridian's cost (USD 90) and ask Pacific Star to confirm the correction in writing."},
    {"id": "D-10", "date": E("S-016", 28), "by": LEAD, "incident": "I-10", "shipment": "S-016",
     "summary": "Dispute the detention invoice with Hanseatic for 2 of the 6 days (terminal closure); bill Rapid Haul for the remaining 4 days."},
    {"id": "D-11", "date": "2026-04-28", "by": OPS_MGR, "incident": None, "shipment": None,
     "summary": "From Q2 2026 all reefer imports for Blue Harbor Foods are routed via Kim & Park's Busan office only."},
    {"id": "D-12", "date": "2026-06-19", "by": LEAD, "incident": None, "shipment": None,
     "summary": "Daniel Cho takes over all open Daehan Precision and Seoul Medica files from Jiwoo Han effective 1 July 2026."},
]
DEC = {d_["id"]: d_ for d_ in DECISIONS}

TASKS = [
    {"id": "T-01", "status": "open", "owner": JUNIOR, "shipment": "S-017", "incident": "I-07", "due": "2026-07-10",
     "summary": "Collect surveyor report and submit formal claim to Orient Container Line."},
    {"id": "T-02", "status": "open", "owner": JUNIOR, "shipment": "S-016", "incident": "I-10", "due": "2026-07-03",
     "summary": "Follow up Hanseatic D&D desk on detention dispute for the two terminal-closure days."},
    {"id": "T-03", "status": "open", "owner": ACCT, "shipment": "S-016", "incident": "I-10", "due": "2026-07-15",
     "summary": "Issue debit note to Rapid Haul Logistics for 4 days detention."},
    {"id": "T-04", "status": "open", "owner": JUNIOR, "shipment": "S-024", "incident": None, "due": "2026-07-01",
     "summary": "Obtain signed POD for S-024 and hand cost sheet to Accounting."},
    {"id": "T-05", "status": "open", "owner": JUNIOR, "shipment": "S-025", "incident": None, "due": "2026-06-26",
     "summary": "Send Arrival Notice and request documents for S-025 (ETA 29 June)."},
    {"id": "T-06", "status": "open", "owner": OPS_MGR, "shipment": None, "incident": "I-05", "due": "2026-07-31",
     "summary": "Q2 vendor review: discuss recorded non-conformance with Harbor Gate Customs Brokerage."},
    {"id": "T-07", "status": "open", "owner": JUNIOR, "shipment": None, "incident": None, "due": "2026-07-07",
     "summary": "Update the SOP-03 free-time table after Pacific Star's new reefer free time (5 days from July)."},
    {"id": "T-08", "status": "open", "owner": DOCS, "shipment": "S-021", "incident": "I-09", "due": "2026-07-04",
     "summary": "File Pacific Star's written manifest-correction confirmation in the S-021 document register."},
    {"id": "T-09", "status": "done", "owner": OP, "shipment": "S-009", "incident": "I-03", "due": workday(plus(DEC["D-03"]["date"], 3)),
     "done": workday(plus(DEC["D-03"]["date"], 2)), "summary": "Send 50/50 demurrage settlement letter to Blue Harbor Foods."},
    {"id": "T-10", "status": "done", "owner": OP, "shipment": "S-015", "incident": "I-06", "due": workday(plus(DEC["D-06"]["date"], 7)),
     "done": workday(plus(DEC["D-06"]["date"], 4)), "summary": "Circulate SOP-02 2.4 reminder to the import team."},
]
TASK = {t["id"]: t for t in TASKS}

# ---------------------------------------------------------------------------
# Thread scripts (beats). Each beat = one e-mail the LLM will write.
# ---------------------------------------------------------------------------
THREADS: list[dict] = []
_tid = 0
_mid = 0


def new_thread(subject: str, shipment: str | None, kind: str, incident: str | None = None) -> dict:
    global _tid
    _tid += 1
    t = {"id": f"TH-{_tid:03d}", "subject": subject, "shipment": shipment, "kind": kind,
         "incident": incident, "beats": []}
    THREADS.append(t)
    return t


def beat(th: dict, dt: str, frm: str, to: list[str], intent: str, cc: list[str] | None = None,
         attachments: list[str] | None = None, decisions: list[str] | None = None,
         tasks: list[str] | None = None) -> None:
    """One e-mail. `decisions`/`tasks` list the ids this mail evidences."""
    global _mid
    _mid += 1
    th["beats"].append({
        "id": f"M-{_mid:04d}", "date": workday(dt), "from": frm, "to": to, "cc": cc or [],
        "intent": intent, "attachments": attachments or [],
        "decisions": decisions or [], "tasks": tasks or [],
    })


def subj_of(sid: str) -> str:
    s = SH[sid]
    return f"[{s['hbl']} / {s['mbl']}] {s['cargo'].title()} - {s['vessel']} {s['voyage']}"


for s in SHIPMENTS:
    sid, eta = s["id"], s["eta"]
    car, brk, cns = s["carrier"], s["broker"], s["consignee"]
    car_cs, brk_main, cns_ct = CARRIER_CS[car], BROKER_MAIN[brk], CONSIGNEE_CONTACT[cns]
    extra_docs = (", health certificate" if cns == "ORG-BHF" else "") + (", import licence copy" if cns == "ORG-SMS" else "")

    # Thread A: pre-alert -> arrival notice -> documents
    th = new_thread(f"Arrival Notice {subj_of(sid)}", sid, "arrival_notice")
    beat(th, plus(eta, -9), car_cs, [OP],
         f"Carrier pre-alert for {s['vessel']} {s['voyage']}: MBL {s['mbl']}, container {s['container']} ({s['container_type']}), ETA Busan {eta}.")
    if s["status"] == "pre_arrival":
        continue   # T-05: Arrival Notice for S-025 is still to be sent at handover
    beat(th, plus(eta, -7), OP, [cns_ct],
         f"Arrival Notice: quote MBL, HBL, container, vessel/voyage, ETA {eta}, free time {s['free_time_days']} days (expires {expiry(sid)}); request commercial invoice, packing list{extra_docs}.",
         cc=[brk_main], attachments=[f"ArrivalNotice_{s['hbl']}.pdf"])
    beat(th, plus(eta, -5), cns_ct, [OP],
         "Consignee sends the requested documents and asks for the expected clearance date.",
         attachments=[f"CI_PL_{s['hbl']}.pdf"])

    # Thread B: broker instruction -> clearance -> D/O
    th = new_thread(f"Customs clearance {subj_of(sid)}", sid, "clearance")
    beat(th, plus(eta, -4), OP, [brk_main],
         f"Instruct broker: full document set attached, proposed HS code {s['hs_code']}, please file on arrival; ETA {eta}.",
         cc=[BROKER_ASSIST[brk]], attachments=[f"Docs_{s['hbl']}.zip"])
    beat(th, plus(eta, -3), brk_main, [OP],
         f"Broker confirms receipt, agrees HS code {s['hs_code']}, estimates duty and VAT.")
    if s["incident"] in ("I-01", "I-02", "I-04", "I-05", "I-06", "I-08", "I-09"):
        continue   # the incident thread carries the rest of this file's story
    beat(th, plus(eta, 1), brk_main, [OP], "Broker reports customs release obtained; declaration number quoted.")
    beat(th, plus(eta, 1), OP, [car_cs],
         f"Request D/O for container {s['container']}; telex release confirmed on {plus(eta, -2)} and filed.")
    beat(th, plus(eta, 2), car_cs, [OP], "Carrier issues D/O; reminds of free-time expiry date.",
         attachments=[f"DO_{s['mbl']}.pdf"])

    # Thread C: trucking -> delivery -> POD -> cost sheet
    th = new_thread(f"Trucking & delivery {subj_of(sid)}", sid, "delivery")
    beat(th, plus(eta, 2), OP, [TRUCK_MGR],
         f"Book trucking for container {s['container']} from Busan New Port to consignee, pick-up {plus(eta, 3)}, delivery slot next morning.",
         cc=[cns_ct])
    if s["status"] == "awaiting_pod":
        beat(th, plus(eta, 2), TRUCK_DISP, [OP], "Dispatcher confirms truck and driver; asks for D/O copy.")
        continue   # T-04: POD for S-024 still outstanding
    if s["incident"] == "I-07":
        continue   # delivery of S-017 ends in the damage claim thread
    beat(th, plus(eta, 4), cns_ct, [OP], "Consignee confirms cargo received in good order.")
    beat(th, plus(eta, 5), OP, [ACCT], f"Cost sheet for {s['hbl']} attached for invoicing; signed POD attached.",
         cc=[DOCS], attachments=[f"POD_{s['hbl']}.pdf", f"CostSheet_{s['hbl']}.xlsx"])


def inc_thread(inc_id: str, prefix: str) -> tuple[dict, dict, dict]:
    inc = INC[inc_id]
    s = SH[inc["shipment"]]
    th = new_thread(f"{prefix} [{s['hbl']} / {s['mbl']}] {s['cargo'].title()}", s["id"], "incident", inc_id)
    return inc, s, th


# I-01 vessel delay (compliant)
i, s, th = inc_thread("I-01", "Vessel delay 6 days -")
beat(th, i["detected"], CARRIER_CS[s["carrier"]], [OP], "Carrier advises vessel delayed 6 days due to port congestion at Shanghai; revised ETA.")
beat(th, i["detected"], OP, [CONSIGNEE_CONTACT[s["consignee"]]], "Inform consignee of the 6-day delay the same day; revised ETA; offer daily updates.", cc=[LEAD])
beat(th, i["detected"], OP, [LEAD], "Escalate as Severity 1 (delivery delay > 5 days, consignee production risk).")
beat(th, DEC["D-01"]["date"], LEAD, [OP], "Decision D-01: accept the delay, no re-routing; daily ETA update to consignee.", cc=[OPS_MGR], decisions=["D-01"])
beat(th, plus(DEC["D-01"]["date"], 1), CONSIGNEE_MGR[s["consignee"]], [OP], "Consignee manager thanks for transparency, asks for priority clearance on arrival.")
beat(th, plus(s["eta"], 7), BROKER_MAIN[s["broker"]], [OP], "Release obtained the day after the delayed arrival; D/O and trucking arranged as usual.")

# I-02 missing docs (compliant)
i, s, th = inc_thread("I-02", "Health certificate missing -")
exp = expiry(s["id"])
beat(th, i["detected"], BROKER_MAIN[s["broker"]], [OP], "Broker cannot file: health certificate for frozen shrimp missing from the document set.")
beat(th, i["detected"], OP, [CONSIGNEE_CONTACT[s["consignee"]]], f"Chase the health certificate in writing; warn that free time expires on {exp}.")
beat(th, plus(exp, -4), OP, [CONSIGNEE_CONTACT[s["consignee"]]], "Formal written reminder 3 working days before free-time expiry (SOP-03 3.2).", cc=[LEAD])
beat(th, DEC["D-02"]["date"], CONSIGNEE_CONTACT[s["consignee"]], [OP], "Consignee sends the health certificate.", attachments=[f"HealthCert_{s['hbl']}.pdf"])
beat(th, DEC["D-02"]["date"], OP, [BROKER_MAIN[s["broker"]]], "Decision D-02: file today using the priority slot; certificate attached.", attachments=[f"HealthCert_{s['hbl']}.pdf"], decisions=["D-02"])
beat(th, plus(DEC["D-02"]["date"], 1), BROKER_MAIN[s["broker"]], [OP], "Release obtained within free time; D/O requested.")

# I-03 demurrage (non-compliant: late extension, late escalation)
i, s, th = inc_thread("I-03", "Pick-up hold request / free time -")
exp = expiry(s["id"])
beat(th, plus(s["eta"], 1), CONSIGNEE_CONTACT[s["consignee"]], [OP], "Consignee asks to hold pick-up about a week because their warehouse is full.")
beat(th, plus(s["eta"], 1), OP, [CONSIGNEE_CONTACT[s["consignee"]]], f"Acknowledge; note free time expires {exp}; will try for an extension.")
beat(th, exp, OP, [CARRIER_DND[s["carrier"]]], "Request a 5-day free-time extension (sent on the expiry day itself).")
beat(th, i["detected"], CARRIER_DND[s["carrier"]], [OP], "Carrier declines: extension requests must be received before expiry; demurrage accrues from today.")
beat(th, i["detected"], OP, [CONSIGNEE_CONTACT[s["consignee"]]], "Inform consignee that demurrage is now accruing; push for pick-up.")
beat(th, plus(i["detected"], 4), CARRIER_DND[s["carrier"]], [OP], "Demurrage invoice: 4 days, USD 1,120.", attachments=[f"DND_Invoice_{s['mbl']}.pdf"])
beat(th, i["escalated"], OP, [LEAD], "Escalate the demurrage to Team Lead (5 days after the carrier's refusal).")
beat(th, DEC["D-03"]["date"], LEAD, [OP], "Decision D-03: Meridian absorbs 50% (USD 560), consignee billed the rest; reason: late extension request.", cc=[OPS_MGR, "P-007"], decisions=["D-03"])
beat(th, TASK["T-09"]["done"], OP, [CONSIGNEE_MGR[s["consignee"]]], "Settlement letter: 50/50 split of demurrage (task T-09 done).", cc=[LEAD], attachments=[f"Settlement_{s['hbl']}.pdf"], tasks=["T-09"])

# I-04 customs inspection (compliant)
i, s, th = inc_thread("I-04", "Customs physical inspection -")
beat(th, i["detected"], BROKER_MAIN[s["broker"]], [OP], "Customs selected the container for physical inspection; import licence number queried.")
beat(th, i["detected"], OP, [CONSIGNEE_CONTACT[s["consignee"]]], "Inform consignee the same day; ask for the import licence copy.", cc=[LEAD])
beat(th, DEC["D-04"]["date"], OP, [BROKER_MAIN[s["broker"]], CONSIGNEE_CONTACT[s["consignee"]]], "Decision D-04: operator attends the inspection with the broker; consignee sends the licence directly to customs.", decisions=["D-04"])
beat(th, plus(DEC["D-04"]["date"], 1), BROKER_MAIN[s["broker"]], [OP], "Inspection passed, release obtained; D/O requested.")

# I-05 HS code dispute (non-compliant)
i, s, th = inc_thread("I-05", "HS code filed differs from history -")
beat(th, plus(s["eta"], 1), BROKER_MAIN[s["broker"]], [OP], "Release obtained; declaration filed under HS 8482.99 (reclassified).")
beat(th, i["detected"], CONSIGNEE_CONTACT[s["consignee"]], [OP], "Consignee notices the declaration used HS 8482.99 instead of their usual 8482.10; duty differs by about USD 380; asks why nobody confirmed.")
beat(th, i["detected"], OP, [BROKER_MAIN[s["broker"]]], "Ask broker why the HS code was changed without written confirmation.", cc=[LEAD])
beat(th, plus(i["detected"], 1), BROKER_MAIN[s["broker"]], [OP], "Broker explains the classification ruling and apologises for not confirming first.")
beat(th, DEC["D-05"]["date"], LEAD, [OP, BROKER_MAIN[s["broker"]]], "Decision D-05: keep the filed code; record a non-conformance against the broker for filing without written confirmation.", cc=[OPS_MGR], decisions=["D-05"])

# I-06 D/O before release confirmation (non-compliant)
i, s, th = inc_thread("I-06", "D/O refused - telex release not confirmed -")
beat(th, plus(s["eta"], -1), OP, [CARRIER_CS[s["carrier"]]], "Request D/O for arrival tomorrow (telex release not yet confirmed in the file).")
beat(th, i["detected"], CARRIER_CS[s["carrier"]], [OP], "Carrier refuses D/O: no telex release received from origin yet.")
beat(th, i["detected"], OP, [SHIPPER_CT], "Ask shipper to arrange telex release urgently.", cc=[CONSIGNEE_CONTACT[s["consignee"]]])
beat(th, i["detected"], OP, [CONSIGNEE_CONTACT[s["consignee"]]], "Inform consignee of the expected 2-day release delay.")
beat(th, i["detected"], SHIPPER_CT, [OP], "Shipper confirms telex release instructed to carrier; carrier will confirm in 1-2 days.")
beat(th, i["escalated"], OP, [LEAD], "Escalate: 2 days demurrage (USD 560) caused by the premature D/O request.")
beat(th, DEC["D-06"]["date"], LEAD, [OP], "Decision D-06: absorb USD 560 as Meridian error; re-train the team on SOP-02 2.4.", cc=[OPS_MGR], decisions=["D-06"])
beat(th, TASK["T-10"]["done"], OP, [LEAD, JUNIOR, DOCS], "SOP-02 2.4 reminder circulated to the team (task T-10 done).", tasks=["T-10"])

# I-07 cargo damage claim (compliant)
i, s, th = inc_thread("I-07", "Wet damage at delivery - claim -")
beat(th, i["detected"], CONSIGNEE_CONTACT[s["consignee"]], [OP], "Consignee reports wet damage on 14 cartons at delivery; photos attached; delivery receipt annotated.", attachments=["Damage_photos.zip"])
beat(th, i["detected"], OP, [LEAD], "Escalate Severity 1: claim likely over USD 10,000.", cc=[OPS_MGR])
beat(th, plus(i["detected"], 1), OP, [CARRIER_CS[s["carrier"]]], "Formal claim notice to carrier within 3 days of delivery; reserve all rights.", cc=[LEAD], attachments=[f"ClaimNotice_{s['hbl']}.pdf"])
beat(th, DEC["D-07"]["date"], OPS_MGR, [OP, LEAD], "Decision D-07: appoint an independent surveyor; pursue the claim against the carrier; weekly update to consignee.", decisions=["D-07"])
beat(th, plus(DEC["D-07"]["date"], 7), OP, [CONSIGNEE_CONTACT[s["consignee"]]], "Weekly claim update: surveyor appointed, inspection scheduled.")
beat(th, "2026-06-22", OP, [JUNIOR], "Handover of the claim file: survey done, report pending - task T-01 assigned to Daniel.", cc=[LEAD], tasks=["T-01"])

# I-08 urgent clearance (non-compliant)
i, s, th = inc_thread("I-08", "URGENT same-day clearance - line stop -")
beat(th, i["detected"], CONSIGNEE_MGR[s["consignee"]], [OP], "Consignee's production line will stop tomorrow; needs the cargo today.", cc=[SALES])
beat(th, i["detected"], OP, [BROKER_MAIN[s["broker"]]], "Ask broker to expedite today; accept expedite fee USD 150 verbally (Decision D-08).", decisions=["D-08"])
beat(th, i["detected"], OP, [TRUCK_MGR], "Book same-day pick-up with Rapid Haul.", cc=[CONSIGNEE_CONTACT[s["consignee"]]])
beat(th, i["detected"], BROKER_MAIN[s["broker"]], [OP], "Release obtained 15:40; D/O to follow; expedite fee USD 150 will be invoiced.")
beat(th, plus(i["detected"], 1), SALES, [OP], "Sales manager thanks the team; asks that the expedite fee be covered by Meridian as goodwill.", cc=[LEAD])

# I-09 manifest discrepancy (compliant)
i, s, th = inc_thread("I-09", "Manifest shows 18 pallets vs B/L 20 -")
beat(th, i["detected"], BROKER_MAIN[s["broker"]], [OP], "Customs system shows manifest quantity 18 pallets vs 20 on the B/L; filing blocked.")
beat(th, i["detected"], OP, [CARRIER_CS[s["carrier"]]], "Request manifest correction to 20 pallets (vessel already arrived).", cc=[LEAD])
beat(th, i["detected"], OP, [CONSIGNEE_CONTACT[s["consignee"]]], "Inform consignee of the discrepancy and likely 1-2 day delay.")
beat(th, i["escalated"], OP, [LEAD], "Escalate late manifest correction per SOP-02 3.2.")
beat(th, DEC["D-09"]["date"], LEAD, [OP], "Decision D-09: file the correction at Meridian's cost USD 90; ask carrier for written confirmation.", decisions=["D-09"])
beat(th, plus(DEC["D-09"]["date"], 1), BROKER_MAIN[s["broker"]], [OP], "Correction accepted, release obtained.")
beat(th, "2026-06-22", OP, [DOCS], "Task T-08: file the carrier's written confirmation in the register once received.", tasks=["T-08"])

# I-10 detention (non-compliant)
i, s, th = inc_thread("I-10", "Detention invoice - late empty return -")
beat(th, i["detected"], CARRIER_DND[s["carrier"]], [OP], "Detention invoice: empty returned 6 days after free time, USD 900.", attachments=[f"Detention_{s['mbl']}.pdf"])
beat(th, i["detected"], OP, [TRUCK_MGR], "Ask trucker why the empty was returned late; no return date was ever confirmed in writing.")
beat(th, plus(i["detected"], 1), TRUCK_MGR, [OP], "Trucker: the terminal was closed 2 days for a typhoon; the other days were a dispatch backlog.")
beat(th, i["consignee_informed"], OP, [CONSIGNEE_CONTACT[s["consignee"]]], "Inform consignee of the detention charge (2 days after detection).")
beat(th, DEC["D-10"]["date"], LEAD, [OP], "Decision D-10 (Team Lead found out from the May KPI): dispute 2 days with carrier (terminal closure), bill Rapid Haul the other 4 days. Notes the issue was never escalated within 48h.", cc=[OPS_MGR, ACCT], decisions=["D-10"])
beat(th, "2026-06-23", OP, [JUNIOR], "Handover: task T-02 (carrier dispute) to Daniel.", cc=[LEAD], tasks=["T-02"])
beat(th, "2026-06-23", OP, [ACCT], "Handover: task T-03 (debit note to Rapid Haul for 4 days) to Junho.", cc=[LEAD], tasks=["T-03"])

# Non-shipment threads
th = new_thread("Reefer imports for Blue Harbor Foods - broker routing from Q2", None, "policy")
beat(th, "2026-04-27", OPS_MGR, [LEAD, OP, JUNIOR], "Proposal: route all Blue Harbor reefer imports via Kim & Park's Busan office only.", cc=["P-021"])
beat(th, DEC["D-11"]["date"], OPS_MGR, [LEAD, OP, JUNIOR, "P-021"], "Decision D-11 confirmed: effective Q2 2026.", decisions=["D-11"])

th = new_thread("Pacific Star - reefer free time change from July 2026", None, "vendor")
beat(th, "2026-06-16", "P-018", [OP], "Pacific Star announces reefer free time changes to 5 days from 1 July 2026.")
beat(th, "2026-06-17", OP, [JUNIOR], "Task T-07: update the SOP-03 free-time table after the change.", cc=[LEAD], tasks=["T-07"])

th = new_thread("Monthly import KPI - May 2026", None, "report")
beat(th, "2026-06-03", OP, [LEAD, OPS_MGR], "May KPI: files closed, demurrage/detention incidents (S-009, S-015 settled; S-016 detention invoice open), broker lead times.", attachments=["Import_KPI_2026-05.xlsx"])

th = new_thread("Q2 vendor review - Harbor Gate non-conformance", None, "vendor")
beat(th, "2026-06-18", OPS_MGR, [LEAD, OP], "Q2 review agenda: discuss the recorded HS-code non-conformance (I-05) with Harbor Gate (task T-06).", tasks=["T-06"])

th = new_thread("Handover of import files - Jiwoo Han leaving 30 June", None, "handover")
beat(th, DEC["D-12"]["date"], LEAD, [OP, JUNIOR, OPS_MGR], "Decision D-12: Daniel Cho takes over all open Daehan Precision and Seoul Medica files from 1 July.", decisions=["D-12"])
beat(th, "2026-06-24", OP, [JUNIOR], "Handover summary: open tasks T-01..T-08, contacts per carrier/broker, watch-outs (telex release, free-time reminders, written empty-return dates).", cc=[LEAD], attachments=["Handover_Notes_JH.docx"], tasks=["T-04", "T-05"])
beat(th, "2026-06-25", OP, [JUNIOR], "S-025 pre-arrival: Arrival Notice still to be sent (task T-05); S-024 POD still outstanding (task T-04).", tasks=["T-04", "T-05"])

# ---------------------------------------------------------------------------
# Assemble + write
# ---------------------------------------------------------------------------
WORLD = {
    "meta": {"company": COMPANY, "domain": DOMAIN, "period_start": PERIOD_START.isoformat(),
             "period_end": PERIOD_END.isoformat(), "predecessor": OP, "successor": JUNIOR,
             "seed": SEED, "generated_by": "src/gen_world.py"},
    "organizations": ORGS,
    "people": PEOPLE,
    "shipments": SHIPMENTS,
    "sops": SOPS,
    "incidents": INCIDENTS,
    "decisions": DECISIONS,
    "tasks": TASKS,
    "threads": THREADS,
}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        yaml.safe_dump(WORLD, f, sort_keys=False, allow_unicode=True, width=120)
    n_mail = sum(len(t["beats"]) for t in THREADS)
    n_clause = sum(len(sec["clauses"]) for sop in SOPS for sec in sop["sections"])
    print(f"wrote {OUT}")
    print(f"orgs={len(ORGS)} people={len(PEOPLE)} shipments={len(SHIPMENTS)} incidents={len(INCIDENTS)} "
          f"decisions={len(DECISIONS)} tasks={len(TASKS)} sops={len(SOPS)} clauses={n_clause} "
          f"threads={len(THREADS)} mails={n_mail}")


if __name__ == "__main__":
    main()
