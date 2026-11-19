"""Relate store access - the one place that knows where the graph lives.

The whole knowledge graph is a single embedded Relate file (data/handover.rl),
so a reviewer can open it with `relate shell data/handover.rl` or
`relate serve --bolt 7687 --database data/handover.rl` without any server.

Usage:
    from store import connect, init_schema, run
    conn = connect()                 # read/write
    conn = connect(read_only=True)   # retrieval
"""
from __future__ import annotations

import sys
from pathlib import Path

import relate

sys.path.insert(0, str(Path(__file__).resolve().parent))
import schema  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / schema.STORE_PATH


def connect(read_only: bool = False, path: Path | str = STORE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return relate.connect(str(path), read_only=read_only)


def run(conn, cypher: str, params: dict | None = None) -> list[dict]:
    return conn.execute(cypher, params or {}).fetchall()


def init_schema(conn) -> None:
    """Idempotent: constraints + vector indexes from schema.py."""
    for ddl in schema.CONSTRAINTS:
        conn.execute(ddl)
    for name, (label, prop) in schema.VECTOR_INDEXES.items():
        conn.execute(schema.vector_index_ddl(name, label, prop))


def counts(conn) -> dict[str, int]:
    rows = run(conn, "MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS c ORDER BY l")
    return {r["l"]: r["c"] for r in rows}


if __name__ == "__main__":
    c = connect()
    init_schema(c)
    print("store:", STORE)
    print("indexes:", [r["name"] for r in run(c, "SHOW INDEXES YIELD name RETURN name")])
    print("counts:", counts(c))
    c.close()
