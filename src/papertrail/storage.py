"""SQLite checkpoints and atomic artifacts; the last successful node survives a crash."""

import json
import os
import re
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

from papertrail.errors import PaperTrailError
from papertrail.schema import RunState, now


def atomic_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class Store:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.database = root / "sessions.sqlite3"
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, updated TEXT, state TEXT)"
            )

    @contextmanager
    def exclusive(self):
        try:
            with FileLock(str(self.root / "writer.lock"), timeout=0):
                yield
        except Timeout as exc:
            raise PaperTrailError(
                "Another PaperTrail operation is running in this data directory. Wait for it to finish or use --data-dir."
            ) from exc

    def run_dir(self, run_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{12}", run_id):
            raise PaperTrailError(
                "Invalid session ID. Use `papertrail sessions` to list saved sessions."
            )
        path = self.root / "runs" / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, state: RunState) -> None:
        state.updated_at = now()
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?, ?)",
                (state.id, state.updated_at, state.model_dump_json()),
            )

    def load(self, run_id: str) -> RunState:
        with closing(sqlite3.connect(self.database)) as db, db:
            row = db.execute("SELECT state FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise PaperTrailError(f"Session {run_id!r} was not found. Use `papertrail sessions`.")
        return RunState.model_validate_json(row[0])

    def recent(self) -> list[RunState]:
        with closing(sqlite3.connect(self.database)) as db, db:
            rows = db.execute("SELECT state FROM runs ORDER BY updated DESC LIMIT 25").fetchall()
        return [RunState.model_validate_json(row[0]) for row in rows]
