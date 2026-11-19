"""Step 1a: turn the mail beats in world/world.yaml into English e-mails.

The LLM writes prose only. Every fact (who, when, which shipment, amounts,
decisions) is fixed by the beat; the model is told not to invent any.
Internal ground-truth ids (D-03, T-09, I-05...) must not appear in the mails -
that is what the GraphRAG pipeline later has to recover.

Outputs (data/mail/, gitignored):
    json/TH-xxx.json   one file per thread (resume cache)
    eml/M-xxxx.eml     one RFC-822 message per beat, with stub attachments
    mailbox.jsonl      one record per mail (headers + body)
    _usage.json        token usage / cost

Run:  uv run python src/gen_mail.py [--limit N] [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import format_datetime
from pathlib import Path

import anthropic
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import schema  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)  # shell may hold a stale ANTHROPIC_API_KEY
W = yaml.safe_load((ROOT / "world" / "world.yaml").read_text(encoding="utf-8"))
P = {p["id"]: p for p in W["people"]}
ORG = {o["id"]: o for o in W["organizations"]}
SH = {s["id"]: s for s in W["shipments"]}
OUT = ROOT / "data" / "mail"
KST = timezone(timedelta(hours=9))

# Opus 5.5 list price per 1M tokens (anthropic.com/pricing, 2026-11-19)
PRICE_IN, PRICE_OUT = 4.0, 20.0


class Mail(BaseModel):
    beat_id: str
    subject: str
    body: str


class ThreadMails(BaseModel):
    mails: list[Mail]


SYSTEM = f"""You write realistic internal and external business e-mails for {W['meta']['company']}, a Korean sea-freight forwarder in Busan, and for its carriers, customs brokers, consignees, truckers and a German shipper. The year is 2026.

Rules:
- Write one e-mail per beat, in the order given. Each beat says who writes to whom, on which date, with what intent and attachments. Turn the intent into natural prose (60-180 words). Write the way a busy logistics professional writes: short paragraphs, concrete, polite, no fluff.
- Every fact in the intent must appear in the mail (dates, amounts in USD, container/B/L numbers, SOP clause references such as "SOP-03 section 4.2", counts, who decided what). Do NOT add facts that are not in the intent or shipment card: no new dates, amounts, names, documents or events.
- Never write internal record ids such as D-03, T-09, I-05, TH-012, M-0045, P-001 or ORG-MFC in the mail text. Refer to people by name, decisions by content, tasks by what has to be done.
- Shipment and SOP ids (S-001, SOP-03, HBL/MBL/container numbers) MAY appear.
- Later mails in a thread are replies: they can briefly refer back to earlier mails, use "Re:" subjects, and they must stay consistent with what was said before.
- Sign off with the sender's name, title and company. Do not write address headers (From/To/Date) in the body.
- Subjects: keep the thread subject; replies prefix "Re:". Do not include ids other than HBL/MBL.
"""


def who(pid: str) -> str:
    p = P[pid]
    return f"{p['name']} ({p['title']}, {ORG[p['org']]['name']}, {p['email']})"


def shipment_card(sid: str | None) -> str:
    if not sid:
        return "(no specific shipment)"
    s = SH[sid]
    card = (f"{sid}: HBL {s['hbl']}, MBL {s['mbl']}, container {s['container']} ({s['container_type']}), "
            f"vessel {s['vessel']} voy {s['voyage']}, {s['pol_name']} -> Busan, ETA {s['eta']}, "
            f"free time {s['free_time_days']} days, cargo {s['cargo']} (HS {s['hs_code']}), "
            f"carrier {ORG[s['carrier']]['name']}, broker {ORG[s['broker']]['name']}, "
            f"consignee {ORG[s['consignee']]['name']}")
    if s.get("shipper"):
        card += f", shipper {ORG[s['shipper']]['name']}"
    return card


def thread_prompt(th: dict) -> str:
    people = sorted({b["from"] for b in th["beats"]} | {r for b in th["beats"] for r in b["to"] + b["cc"]})
    lines = [f"Thread subject: {th['subject']}", f"Shipment: {shipment_card(th['shipment'])}", "Participants:"]
    lines += [f"  - {who(p)}" for p in people]
    lines += ["", "Beats:"]
    for b in th["beats"]:
        line = (f"- beat_id {b['id']} | date {b['date']} | from {P[b['from']]['name']} "
                f"| to {', '.join(P[r]['name'] for r in b['to'])}")
        if b["cc"]:
            line += f" | cc {', '.join(P[r]['name'] for r in b['cc'])}"
        if b["attachments"]:
            line += f" | attachments {', '.join(b['attachments'])}"
        lines.append(line)
        lines.append(f"    intent: {b['intent']}")
    lines += ["", f"Return exactly {len(th['beats'])} mails with these beat_ids, in order."]
    return "\n".join(lines)


def write_thread(client: anthropic.Anthropic, th: dict) -> dict:
    cache = OUT / "json" / f"{th['id']}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    resp = client.messages.parse(
        model=schema.LLM_MODEL,
        max_tokens=12000,
        system=SYSTEM,
        messages=[{"role": "user", "content": thread_prompt(th)}],
        output_format=ThreadMails,
    )
    mails = {m.beat_id: m for m in resp.parsed_output.mails}
    missing = [b["id"] for b in th["beats"] if b["id"] not in mails]
    if missing:
        raise RuntimeError(f"{th['id']}: model skipped beats {missing}")
    rec = {
        "thread": th["id"],
        "usage": {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens},
        "mails": [{"beat_id": b["id"], "subject": mails[b["id"]].subject, "body": mails[b["id"]].body}
                  for b in th["beats"]],
    }
    cache.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec


def addr(pid: str) -> str:
    return f"{P[pid]['name']} <{P[pid]['email']}>"


def msgid(mid: str) -> str:
    return f"<{mid}@{W['meta']['domain']}>"


MIME = {
    "pdf": ("application", "pdf"),
    "zip": ("application", "zip"),
    "xlsx": ("application", "vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    "docx": ("application", "vnd.openxmlformats-officedocument.wordprocessingml.document"),
}


def build_eml(b: dict, mail: dict, when: datetime, parent: str | None, refs: list[str]) -> EmailMessage:
    msg = EmailMessage()
    msg["Message-ID"] = msgid(b["id"])
    msg["Date"] = format_datetime(when)
    msg["From"] = addr(b["from"])
    msg["To"] = ", ".join(addr(r) for r in b["to"])
    if b["cc"]:
        msg["Cc"] = ", ".join(addr(r) for r in b["cc"])
    msg["Subject"] = mail["subject"]
    if parent:
        msg["In-Reply-To"] = msgid(parent)
        msg["References"] = " ".join(msgid(r) for r in refs)
    msg.set_content(mail["body"])
    for fn in b["attachments"]:
        ext = fn.rsplit(".", 1)[-1].lower()
        maintype, subtype = MIME.get(ext, ("application", "octet-stream"))
        msg.add_attachment(f"[placeholder for {fn}]".encode(), maintype=maintype, subtype=subtype, filename=fn)
    if msg.is_multipart():
        msg.set_boundary(f"=={b['id']}==")   # deterministic: re-running the generator must not touch git
    return msg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N threads (smoke test)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    threads = W["threads"][: args.limit] if args.limit else W["threads"]
    client = anthropic.Anthropic()
    (OUT / "json").mkdir(parents=True, exist_ok=True)
    (OUT / "eml").mkdir(parents=True, exist_ok=True)

    recs: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(write_thread, client, th): th for th in threads}
        for f in as_completed(futs):
            th = futs[f]
            rec = f.result()
            recs[th["id"]] = rec
            print(f"{th['id']} ok ({len(rec['mails'])} mails, "
                  f"{rec['usage']['input']}/{rec['usage']['output']} tok)", flush=True)

    # Assemble mailbox: deterministic times, reply chain
    rng = random.Random(W["meta"]["seed"])
    rows = []
    for th in threads:
        rec = recs[th["id"]]
        mails = {m["beat_id"]: m for m in rec["mails"]}
        last_time: datetime | None = None
        chain: list[str] = []
        for b in th["beats"]:
            day = datetime.fromisoformat(b["date"]).replace(tzinfo=KST)
            when = day.replace(hour=rng.randint(8, 17), minute=rng.randint(0, 59))
            if last_time and when <= last_time:
                when = last_time + timedelta(minutes=rng.randint(20, 150))
            last_time = when
            parent = chain[-1] if chain else None
            msg = build_eml(b, mails[b["id"]], when, parent, chain)
            (OUT / "eml" / f"{b['id']}.eml").write_bytes(bytes(msg))
            rows.append({
                "id": b["id"], "message_id": msgid(b["id"]), "thread": th["id"], "date": when.isoformat(),
                "from": P[b["from"]]["email"], "to": [P[r]["email"] for r in b["to"]],
                "cc": [P[r]["email"] for r in b["cc"]], "subject": mails[b["id"]]["subject"],
                "in_reply_to": msgid(parent) if parent else None,
                "attachments": b["attachments"], "body": mails[b["id"]]["body"],
            })
            chain.append(b["id"])
    with (OUT / "mailbox.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tin = sum(r["usage"]["input"] for r in recs.values())
    tout = sum(r["usage"]["output"] for r in recs.values())
    usage = {"model": schema.LLM_MODEL, "threads": len(recs), "mails": len(rows),
             "input_tokens": tin, "output_tokens": tout,
             "cost_usd_list_price": round(tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT, 3)}
    (OUT / "_usage.json").write_text(json.dumps(usage, indent=1), encoding="utf-8")
    print(json.dumps(usage))


if __name__ == "__main__":
    main()
