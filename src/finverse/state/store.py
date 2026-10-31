"""Append-only, versioned state store (SQLite, stdlib only).

Every record is identified by (run_id, as_of, kind, key). Writing the same identity again creates version+1; old
versions are never modified or deleted, so any decision can be replayed exactly as it was logged.
The message-queue question is an implementation choice and is kept out of scope: this store is the shared state."""
from __future__ import annotations
import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS records(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL, as_of TEXT NOT NULL, kind TEXT NOT NULL, key TEXT NOT NULL,
  version INTEGER NOT NULL, created_at TEXT NOT NULL, source_ids TEXT, payload TEXT NOT NULL,
  UNIQUE(run_id, as_of, kind, key, version));
CREATE INDEX IF NOT EXISTS ix_lookup ON records(run_id, as_of, kind, key);
CREATE TRIGGER IF NOT EXISTS no_update BEFORE UPDATE ON records BEGIN SELECT RAISE(ABORT, 'records are immutable'); END;
CREATE TRIGGER IF NOT EXISTS no_delete BEFORE DELETE ON records BEGIN SELECT RAISE(ABORT, 'records are immutable'); END;
"""

KINDS = ("agent_output", "regime", "coordinator", "portfolio", "explanation")


class StateStore:
    def __init__(self, path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.executescript(SCHEMA)

    def put(self, run_id, as_of, kind, key, payload: dict, source_ids=None) -> int:
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind}")
        cur = self.db.execute("SELECT COALESCE(MAX(version),0) FROM records WHERE run_id=? AND as_of=? AND kind=? AND key=?",
                              (run_id, str(as_of), kind, key))
        v = cur.fetchone()[0] + 1
        self.db.execute("INSERT INTO records(run_id,as_of,kind,key,version,created_at,source_ids,payload) VALUES(?,?,?,?,?,?,?,?)",
                        (run_id, str(as_of), kind, key, v, time.strftime("%Y-%m-%dT%H:%M:%S"),
                         json.dumps(source_ids or []), json.dumps(payload, default=float)))
        return v

    def get(self, run_id, as_of, kind, key, version=None):
        q = "SELECT version,created_at,source_ids,payload FROM records WHERE run_id=? AND as_of=? AND kind=? AND key=?"
        args = [run_id, str(as_of), kind, key]
        if version is None:
            q += " ORDER BY version DESC LIMIT 1"
        else:
            q += " AND version=?"; args.append(version)
        r = self.db.execute(q, args).fetchone()
        if r is None:
            return None
        return {"version": r[0], "created_at": r[1], "source_ids": json.loads(r[2]), "payload": json.loads(r[3])}

    def history(self, run_id, as_of, kind, key):
        rows = self.db.execute("SELECT version,created_at,payload FROM records WHERE run_id=? AND as_of=? AND kind=? AND key=? ORDER BY version",
                               (run_id, str(as_of), kind, key)).fetchall()
        return [{"version": v, "created_at": c, "payload": json.loads(p)} for v, c, p in rows]

    def dates(self, run_id, kind):
        return [r[0] for r in self.db.execute("SELECT DISTINCT as_of FROM records WHERE run_id=? AND kind=? ORDER BY as_of", (run_id, kind))]

    def commit(self):
        self.db.commit()

    def close(self):
        self.db.commit(); self.db.close()
