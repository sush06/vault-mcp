"""vault-mcp MCP server. Run directly for local dev with
`python -m vault_mcp.server`, or point Claude Desktop at it per the README."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from . import embeddings
from .search import search as run_search
from .storage import VaultStore

DEFAULT_DB_PATH = Path(os.environ.get("VAULT_MCP_DB", Path.home() / ".vault-mcp" / "vault.db"))

mcp = FastMCP("vault-mcp")
_store: VaultStore | None = None


def get_store() -> VaultStore:
    global _store
    if _store is None:
        _store = VaultStore(DEFAULT_DB_PATH)
    return _store


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


@mcp.tool()
def add_note(title: str, content: str, tags: list[str] | None = None) -> dict:
    """Add a new note to the personal knowledge base.

    Args:
        title: Short title for the note.
        content: The note body (markdown is fine).
        tags: Optional list of tags for filtering/organization.
    """
    store = get_store()
    note_id = store.add_note(title, content, tags)
    vector = embeddings.embed(f"{title}\n{content}")
    store.set_embedding(note_id, vector)
    return {"note_id": note_id, "title": title, "status": "saved"}


@mcp.tool()
def search_notes(query: str, top_k: int = 5) -> list[dict]:
    """Search the knowledge base using hybrid keyword + semantic search.

    Args:
        query: Natural language search query.
        top_k: Maximum number of results to return.
    """
    store = get_store()
    results = run_search(store, query, top_k=top_k, mode="hybrid")
    notes = {n.id: n for n in store.all_notes()}
    return [
        {
            "note_id": r.note_id,
            "title": r.title,
            "snippet": r.snippet,
            "score": round(r.score, 4),
            "matched_by": r.matched_by,
            "created_at": _fmt_ts(notes[r.note_id].created_at) if r.note_id in notes else None,
        }
        for r in results
    ]


@mcp.tool()
def get_note(note_id: int) -> dict:
    """Retrieve the full content of a specific note by ID."""
    store = get_store()
    note = store.get_note(note_id)
    if note is None:
        return {"error": f"No note with id {note_id}"}
    return {
        "note_id": note.id,
        "title": note.title,
        "content": note.content,
        "tags": note.tags.split(",") if note.tags else [],
        "created_at": _fmt_ts(note.created_at),
        "updated_at": _fmt_ts(note.updated_at),
    }


@mcp.tool()
def update_note(note_id: int, title: str, content: str, tags: list[str] | None = None) -> dict:
    """Update an existing note's title, content, and/or tags.

    The note is re-embedded automatically so search reflects the new content.

    Args:
        note_id: ID of the note to update.
        title: New title.
        content: New body (markdown is fine).
        tags: New tag list (replaces existing tags).
    """
    store = get_store()
    updated = store.update_note(note_id, title, content, tags)
    if not updated:
        return {"error": f"No note with id {note_id}"}
    vector = embeddings.embed(f"{title}\n{content}")
    store.set_embedding(note_id, vector)
    return {"note_id": note_id, "title": title, "status": "updated"}


@mcp.tool()
def delete_note(note_id: int) -> dict:
    """Permanently delete a note by ID.

    Args:
        note_id: ID of the note to delete.
    """
    store = get_store()
    if store.get_note(note_id) is None:
        return {"error": f"No note with id {note_id}"}
    store.delete_note(note_id)
    return {"note_id": note_id, "status": "deleted"}


@mcp.tool()
def list_notes(tag: str | None = None, since_days: int | None = None, limit: int = 50) -> list[dict]:
    """Browse notes in the vault, optionally filtered by tag and/or recency.

    Args:
        tag: If provided, only return notes that have this tag.
        since_days: If provided, only return notes created within this many days.
        limit: Maximum number of results to return (default 50).
    """
    store = get_store()
    since = time.time() - since_days * 86400 if since_days is not None else None
    notes = store.list_notes(tag=tag, since=since, limit=limit)
    return [
        {
            "note_id": n.id,
            "title": n.title,
            "tags": n.tags.split(",") if n.tags else [],
            "created_at": _fmt_ts(n.created_at),
            "updated_at": _fmt_ts(n.updated_at),
        }
        for n in notes
    ]


@mcp.tool()
def recall_context(query: str, top_k: int = 3) -> str:
    """Search the knowledge base and return results formatted as context
    text, ready to be dropped into a conversation. Use this when you want
    relevant past notes recalled inline rather than as structured data.
    """
    store = get_store()
    results = run_search(store, query, top_k=top_k, mode="hybrid")
    if not results:
        return "No relevant notes found."
    blocks = [f"### {r.title} (note #{r.note_id})\n{r.snippet}" for r in results]
    return "\n\n".join(blocks)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
