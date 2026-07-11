import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vault_mcp.storage import VaultStore  # noqa: E402


@pytest.fixture
def store(tmp_path):
    s = VaultStore(tmp_path / "test.db")
    yield s
    s.close()


def test_add_and_get_note(store):
    note_id = store.add_note("Test note", "Some content here", ["tag1", "tag2"])
    note = store.get_note(note_id)
    assert note is not None
    assert note.title == "Test note"
    assert note.content == "Some content here"
    assert note.tags == "tag1,tag2"


def test_get_missing_note_returns_none(store):
    assert store.get_note(9999) is None


def test_delete_note(store):
    note_id = store.add_note("Temp", "Delete me", [])
    store.delete_note(note_id)
    assert store.get_note(note_id) is None


def test_keyword_search_finds_exact_term(store):
    store.add_note("Race condition fix", "Fixed a race condition in async pipeline", [])
    store.add_note("Unrelated note", "Something about docker compose", [])
    results = store.keyword_search("race condition")
    assert len(results) == 1
    assert results[0][0] == 1


def test_keyword_search_no_match_returns_empty(store):
    store.add_note("Some note", "Some content", [])
    assert store.keyword_search("zzzznonexistentterm") == []


def test_fts_survives_special_characters(store):
    # Regression test: unescaped punctuation used to break FTS5 MATCH syntax
    store.add_note("C# and .NET notes", "Some content about C# generics", [])
    results = store.keyword_search("C# generics")
    assert len(results) == 1


def test_embedding_roundtrip(store):
    vec = [0.1, 0.2, 0.3, 0.4]
    note_id = store.add_note("Note", "content", [])
    store.set_embedding(note_id, vec)
    all_emb = store.all_embeddings()
    assert note_id in all_emb
    assert all_emb[note_id] == pytest.approx(vec)


def test_all_notes_returns_in_order(store):
    ids = [store.add_note(f"Note {i}", f"content {i}", []) for i in range(3)]
    notes = store.all_notes()
    assert [n.id for n in notes] == ids


def test_update_note_changes_content(store):
    note_id = store.add_note("Original", "old content", ["a"])
    updated = store.update_note(note_id, "Revised", "new content", ["b"])
    assert updated is True
    note = store.get_note(note_id)
    assert note.title == "Revised"
    assert note.content == "new content"
    assert note.tags == "b"


def test_update_note_bumps_updated_at(store):
    import time
    note_id = store.add_note("T", "c", [])
    before = store.get_note(note_id).updated_at
    time.sleep(0.01)
    store.update_note(note_id, "T", "c changed", [])
    after = store.get_note(note_id).updated_at
    assert after > before


def test_update_note_returns_false_for_missing(store):
    assert store.update_note(9999, "x", "y", []) is False


def test_list_notes_returns_all_by_default(store):
    for i in range(3):
        store.add_note(f"Note {i}", "content", [])
    assert len(store.list_notes()) == 3


def test_list_notes_filters_by_tag(store):
    store.add_note("Tagged", "content", ["work"])
    store.add_note("Untagged", "content", [])
    store.add_note("Other tag", "content", ["personal"])
    results = store.list_notes(tag="work")
    assert len(results) == 1
    assert results[0].title == "Tagged"


def test_list_notes_tag_no_partial_match(store):
    # "foo" should not match a note tagged "foobar"
    store.add_note("A", "c", ["foobar"])
    assert store.list_notes(tag="foo") == []


def test_list_notes_filters_by_since(store):
    import time
    cutoff = time.time()
    store.add_note("Old note", "content", [])
    # Manually backdate the first note so it falls before `cutoff`
    store.conn.execute("UPDATE notes SET created_at = ? WHERE id = 1", (cutoff - 100,))
    store.conn.commit()
    store.add_note("New note", "content", [])
    results = store.list_notes(since=cutoff)
    assert len(results) == 1
    assert results[0].title == "New note"


def test_list_notes_respects_limit(store):
    for i in range(10):
        store.add_note(f"Note {i}", "content", [])
    assert len(store.list_notes(limit=3)) == 3


def test_list_notes_ordered_newest_first(store):
    import time
    store.add_note("First", "content", [])
    time.sleep(0.01)
    store.add_note("Second", "content", [])
    results = store.list_notes()
    assert results[0].title == "Second"
