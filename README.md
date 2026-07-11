# vault-mcp

An MCP server for my personal markdown notes vault. Claude (or any MCP client) can save notes and pull them back later by meaning, not just exact wording — "what did I decide about the routing taxonomy" can find a note that never uses those words.

Search is hybrid: keyword (SQLite FTS5 / BM25) plus semantic (local embeddings), combined with Reciprocal Rank Fusion. Pure vector search misses exact-term queries — function names, error strings, jargon. Pure keyword search misses paraphrases ("why did I skip a client-server database" should find a note about picking SQLite over Postgres, even with zero shared words). RRF combines the two using rank position rather than raw score, since BM25 and cosine similarity live on different scales and blending them directly needs fragile manual tuning.

I didn't want to just assume hybrid was better, so there's a small hand-labeled eval set (`eval/eval_set.json`, 29 queries over 25 seeded notes) measuring recall@3 and MRR for keyword-only, vector-only, and hybrid:

| Mode          | Recall@3  | MRR       |
|---------------|-----------|-----------|
| Keyword only  | 0.948     | 0.879     |
| Vector only   | 0.931     | 0.885     |
| **Hybrid**    | **1.000** | **0.937** |

A few queries in the set are deliberately adversarial — pure paraphrases with no shared vocabulary, and exact-jargon queries mixed in with near-duplicate "confusable" notes. An earlier, easier version of this eval set scored 1.0 recall across all three modes, which told me the eval was useless, not that the methods were equivalent. Run it yourself:

```bash
python scripts/seed_demo.py --db /tmp/demo_vault.db
python eval/run_eval.py --db /tmp/demo_vault.db --k 3
```

## How it's put together

Everything lives in one SQLite file — notes, the FTS5 keyword index, and embeddings stored as raw float blobs (cosine similarity computed in Python/numpy, no vector DB needed). One portable file, works fully offline, no API costs.

- `storage.py` — SQLite + FTS5 keyword index
- `embeddings.py` — local sentence-transformers model (`all-MiniLM-L6-v2`)
- `search.py` — fuses keyword + vector results with RRF
- `server.py` — the MCP tool definitions Claude actually calls

## Tools exposed

| Tool              | Description                                                          |
|-------------------|-----------------------------------------------------------------------|
| `add_note`        | Save a note (title, content, tags) — embedded and indexed immediately |
| `search_notes`    | Hybrid search, returns structured results with match provenance       |
| `get_note`        | Fetch full content of a note by ID                                     |
| `update_note`     | Edit an existing note's title/content/tags and re-embed it             |
| `list_notes`      | Browse notes, optionally filtered by tag or recency                    |
| `recall_context`  | Search + format results as ready-to-inject conversation context        |
| `delete_note`     | Remove a note                                                          |

## Quickstart

```bash
git clone <this-repo>
cd vault-mcp
pip install -e .
python scripts/seed_demo.py   # optional: populate with example notes
```

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "vault-mcp": {
      "command": "python",
      "args": ["-m", "vault_mcp.server"],
      "env": { "VAULT_MCP_DB": "/path/to/your/vault.db" }
    }
  }
}
```

## A few decisions worth explaining

**RRF over blending scores.** BM25 and cosine similarity aren't on comparable scales, so averaging them directly means fragile manual tuning. RRF sidesteps that by using rank position instead. See `search.py`.

**SQLite over Postgres or a vector DB.** This is a single-user local tool, not a multi-tenant service. FTS5 already gives BM25-quality keyword search for free, with zero extra infrastructure.

**Local embeddings over an API.** `all-MiniLM-L6-v2` runs fully offline — no per-query cost, no key to manage. It's good enough for note-length text. Swapping in a hosted embedding API later would be a one-line change in `embeddings.py` if I ever need higher quality.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

CI (`.github/workflows/ci.yml`) runs the test suite across Python 3.10–3.12 and re-runs the retrieval eval on every push, so a change that quietly regresses search quality gets caught too, not just one that breaks a unit test.

## What's next

- Sync from Notion/Obsidian as a note source instead of only `add_note`
- Grow the eval set as the vault grows — 29 queries is a start, not a ceiling
- This project's routing-vs-fusion problem is a small piece of something bigger I'm working on: routing natural language queries across heterogeneous backends (SQL, graph, generated code). More on that once PolyQuery is public.
