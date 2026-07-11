"""Storage layer for vault-mcp: notes, an FTS5 keyword index kept in sync
via triggers, and embeddings stored as raw float32 blobs. One SQLite file,
no vector extension needed."""

from __future__ import annotations

import sqlite3
import struct
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Note:
    id: int
    title: str
    content: str
    tags: str  # comma-separated
    created_at: float
    updated_at: float


SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    content     TEXT NOT NULL,
    tags        TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    title, content, tags, content='notes', content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
    INSERT INTO notes_fts(rowid, title, content, tags)
    VALUES (new.id, new.title, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, title, content, tags)
    VALUES ('delete', old.id, old.title, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, title, content, tags)
    VALUES ('delete', old.id, old.title, old.content, old.tags);
    INSERT INTO notes_fts(rowid, title, content, tags)
    VALUES (new.id, new.title, new.content, new.tags);
END;

CREATE TABLE IF NOT EXISTS embeddings (
    note_id     INTEGER PRIMARY KEY REFERENCES notes(id) ON DELETE CASCADE,
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL
);
"""


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _unpack(blob: bytes, dim: int) -> list[float]:
    return list(struct.unpack(f"{dim}f", blob))


class VaultStore:
    """Thin, dependency-free wrapper around the SQLite vault database."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # -- writes ----------------------------------------------------------

    def add_note(self, title: str, content: str, tags: list[str] | None = None) -> int:
        now = time.time()
        tag_str = ",".join(tags or [])
        cur = self.conn.execute(
            "INSERT INTO notes (title, content, tags, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (title, content, tag_str, now, now),
        )
        self.conn.commit()
        return cur.lastrowid

    def set_embedding(self, note_id: int, vector: list[float]) -> None:
        self.conn.execute(
            "INSERT INTO embeddings (note_id, dim, vector) VALUES (?, ?, ?) "
            "ON CONFLICT(note_id) DO UPDATE SET dim=excluded.dim, vector=excluded.vector",
            (note_id, len(vector), _pack(vector)),
        )
        self.conn.commit()

    def update_note(self, note_id: int, title: str, content: str, tags: list[str] | None = None) -> bool:
        now = time.time()
        tag_str = ",".join(tags or [])
        cur = self.conn.execute(
            "UPDATE notes SET title=?, content=?, tags=?, updated_at=? WHERE id=?",
            (title, content, tag_str, now, note_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def delete_note(self, note_id: int) -> None:
        self.conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        self.conn.execute("DELETE FROM embeddings WHERE note_id = ?", (note_id,))
        self.conn.commit()

    # -- reads -------------------------------------------------------------

    def get_note(self, note_id: int) -> Note | None:
        row = self.conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        return Note(**dict(row)) if row else None

    def all_notes(self) -> list[Note]:
        rows = self.conn.execute("SELECT * FROM notes ORDER BY id").fetchall()
        return [Note(**dict(r)) for r in rows]

    def list_notes(
        self,
        tag: str | None = None,
        since: float | None = None,
        limit: int = 50,
    ) -> list[Note]:
        """Return notes ordered by created_at descending.

        Args:
            tag: If set, only return notes whose tag list contains this tag.
            since: If set, only return notes created at or after this Unix timestamp.
            limit: Maximum number of notes to return.
        """
        conditions: list[str] = []
        params: list = []
        if tag:
            # Wrap with commas so "foo" doesn't match "foobar"
            conditions.append("(',' || tags || ',') LIKE ?")
            params.append(f"%,{tag},%")
        if since is not None:
            conditions.append("created_at >= ?")
            params.append(since)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)
        rows = self.conn.execute(
            f"SELECT * FROM notes {where} ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [Note(**dict(r)) for r in rows]

    def all_embeddings(self) -> dict[int, list[float]]:
        rows = self.conn.execute("SELECT note_id, dim, vector FROM embeddings").fetchall()
        return {r["note_id"]: _unpack(r["vector"], r["dim"]) for r in rows}

    def keyword_search(self, query: str, limit: int = 20) -> list[tuple[int, float]]:
        """Returns [(note_id, bm25_score)], best (lowest bm25) first.

        SQLite's bm25() returns *lower is better*; we negate so callers can
        treat every ranking function as 'higher is better'.
        """
        safe_query = _fts_escape(query)
        rows = self.conn.execute(
            "SELECT rowid, bm25(notes_fts) AS score FROM notes_fts "
            "WHERE notes_fts MATCH ? ORDER BY score LIMIT ?",
            (safe_query, limit),
        ).fetchall()
        return [(r["rowid"], -r["score"]) for r in rows]


def _fts_escape(query: str) -> str:
    """Wrap each token in quotes so punctuation/special chars don't break
    FTS5's query syntax; keeps this a plain OR-of-terms match."""
    tokens = [t for t in query.replace('"', " ").split() if t]
    return " OR ".join(f'"{t}"' for t in tokens) or '""'
