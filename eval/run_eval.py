"""Step 5b: run the 40 questions through all three retrievers and grade the answers.

Grading = two signals per answer:
  keyword  : all `must_contain` strings present (case-insensitive)      -> 0/1
  judge    : Claude compares answer with gold -> correct / partial / wrong (structured output)
             correct = all key facts right, nothing contradicting; partial = some facts; wrong = otherwise or "not found".

Outputs: eval/results/answers.jsonl, eval/results/summary.md, eval/results/summary.json
Run:  uv run python eval/run_eval.py [--modes vector,graph,text2cypher] [--only Q17,Q25]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import schema  # noqa: E402
from retrievers import MODES, ask  # noqa: E402
from store import connect  # noqa: E402

OUT = ROOT / "eval" / "results"
PRICE_IN, PRICE_OUT = 4.0, 20.0


class Grade(BaseModel):
    verdict: Literal["correct", "partial", "wrong"]
    reason: str


JUDGE = """You grade answers from a question-answering system against a gold answer written by the person who knows the ground truth.
correct: every key fact asked for (names, dates, amounts, counts, clause numbers, yes/no) matches the gold; extra correct detail is fine.
partial: some key facts match, others are missing or hedged, and nothing contradicts the gold.
wrong: a key fact contradicts the gold, or the answer says the information was not found.
Dates may be written in any format; numbers may be formatted differently (1,120 = 1120). Judge facts, not style."""


def grade(client: anthropic.Anthropic, q: dict, answer: str) -> tuple[Grade, dict]:
    resp = client.messages.parse(
        model=schema.LLM_MODEL, max_tokens=2000, system=JUDGE,
        messages=[{"role": "user", "content": f"Question: {q['question']}\n\nGold answer: {q['gold']}\n\nSystem answer: {answer}"}],
        output_format=Grade)
    return resp.parsed_output, {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--only", default="")
    ap.add_argument("--resume", action="store_true", help="keep answers.jsonl and only run the (question, mode) pairs it lacks")
    args = ap.parse_args()
    modes = args.modes.split(",")
    questions = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    if args.only:
        keep = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in keep]
    OUT.mkdir(parents=True, exist_ok=True)
    conn = connect(read_only=True)
    client = anthropic.Anthropic()

    rows = []
    judge_tokens = {"input": 0, "output": 0}
    out_file = OUT / "answers.jsonl"
    if args.resume and out_file.exists():
        rows = [json.loads(l) for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = {(r["id"], r["mode"]) for r in rows}
    with out_file.open("a" if args.resume else "w", encoding="utf-8") as f:
        for q in questions:
            for mode in modes:
                if (q["id"], mode) in done:
                    continue
                t0 = time.time()
                r = ask(conn, client, q["question"], mode)
                dt = time.time() - t0
                kw = all(k.lower() in r.answer.lower() for k in q["must_contain"])
                g, u = grade(client, q, r.answer)
                judge_tokens = {k: judge_tokens[k] + u[k] for k in judge_tokens}
                row = {"id": q["id"], "type": q["type"], "mode": mode, "question": q["question"], "gold": q["gold"],
                       "answer": r.answer, "keyword_ok": kw, "verdict": g.verdict, "judge_reason": g.reason,
                       "cypher": r.cypher, "error": r.error, "retrieved": r.retrieved, "seconds": round(dt, 1),
                       "tokens": r.usage, "context_chars": len(r.context)}
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"{q['id']} {mode:12s} {g.verdict:8s} kw={int(kw)} {dt:5.1f}s", flush=True)
    conn.close()

    # summary
    score = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}
    by = defaultdict(list)
    for r in rows:
        by[(r["type"], r["mode"])].append(r)
        by[("ALL", r["mode"])].append(r)
    types = ["factual", "aggregate", "multihop", "compliance", "handover", "ALL"]
    lines = ["| question type | n | " + " | ".join(f"{m} correct / partial / score" for m in modes) + " |",
             "|---|---|" + "---|" * len(modes)]
    summary = {}
    for t in types:
        cells = []
        n = 0
        for m in modes:
            rs = by.get((t, m), [])
            n = len(rs)
            c = sum(1 for r in rs if r["verdict"] == "correct")
            p = sum(1 for r in rs if r["verdict"] == "partial")
            s = sum(score[r["verdict"]] for r in rs) / n if n else 0
            cells.append(f"{c} / {p} / {s:.2f}")
            summary[f"{t}/{m}"] = {"n": n, "correct": c, "partial": p, "score": round(s, 3)}
        lines.append(f"| {t} | {n} | " + " | ".join(cells) + " |")
    tin = sum(r["tokens"]["input"] for r in rows)
    tout = sum(r["tokens"]["output"] for r in rows)
    cost = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
    jcost = judge_tokens["input"] / 1e6 * PRICE_IN + judge_tokens["output"] / 1e6 * PRICE_OUT
    lat = {m: round(sum(r["seconds"] for r in rows if r["mode"] == m) / max(1, sum(1 for r in rows if r["mode"] == m)), 1) for m in modes}
    md = ["# Evaluation: 40 questions x 3 retrievers", "",
          f"Model `{schema.LLM_MODEL}`, embeddings `{schema.EMBEDDING_MODEL}`, store `{schema.STORE_PATH}` (Relate).", "",
          "Judge verdicts per question type (correct / partial / mean score where correct=1, partial=0.5).", ""] + lines + [
          "", f"Answering cost (list price): USD {cost:.2f} for {len(rows)} answers; judge: USD {jcost:.2f}.",
          f"Mean latency per answer (s): {lat}", ""]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({"summary": summary, "cost_usd": round(cost, 3), "judge_cost_usd": round(jcost, 3),
                                                    "latency_s": lat}, indent=1), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
