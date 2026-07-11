"""
Seeds a fresh demo vault with a realistic mix of note types so the eval
harness has something meaningful to measure against.

Run: python scripts/seed_demo.py [--db PATH]

Note IDs are deterministic (1, 2, 3, ...) as long as this runs against a
fresh/empty database, which is what eval/eval_set.json assumes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vault_mcp import embeddings  # noqa: E402
from vault_mcp.storage import VaultStore  # noqa: E402

# (title, content, tags) — deliberately mixed lengths/shapes: short facts,
# longer narrative notes, and code-flavored cheat sheets, so the eval
# reflects a realistic vault rather than uniform toy data.
NOTES = [
    ("Race condition fix — AutoDashboard",
     "Fixed a race condition causing a 15% 'Chapter not found' failure rate. "
     "Root cause: async execution pipeline had no dependency resolution between "
     "steps, so downstream tasks sometimes ran before their inputs existed. "
     "Fix: added explicit dependency graph resolution before dispatch. Zero "
     "ordering failures since.",
     ["debugging", "strategy"]),

    ("Interview answer: debugging story",
     "When asked to describe a hard bug: the AutoDashboard race condition. "
     "Intermittent 'Chapter not found' errors, 15% failure rate, traced to "
     "missing dependency ordering in the async pipeline. Framed as: found it "
     "via log correlation, fixed with dependency resolution, verified with "
     "load testing.",
     ["interview"]),

    ("MCP protocol basics",
     "MCP (Model Context Protocol) servers expose tools as typed functions. "
     "The server handles JSON-RPC framing so you just write plain Python "
     "functions with type hints and docstrings. FastMCP wraps this into a "
     "decorator-based API: @mcp.tool() over a function is enough.",
     ["mcp", "reference"]),

    ("MCP server auth patterns",
     "For MCP servers that need auth (calling an internal API, etc), keep "
     "credentials server-side — never expose them as tool parameters the "
     "model could leak into a transcript. Read from environment variables "
     "at server startup instead.",
     ["mcp", "reference"]),

    ("Paper notes: SM3-Text-to-Query",
     "SM3 benchmarks text-to-query generation across SQL, MongoDB query "
     "language, Cypher, and SPARQL. Key limitation for my purposes: it "
     "replicates the same data into every backend, so there's no routing "
     "decision — every question is answerable everywhere. That's the gap "
     "PolyQuery targets.",
     ["research", "polyquery", "papers"]),

    ("Paper notes: Spider 2.0",
     "Spider 2.0 focuses on real-world enterprise text-to-SQL workflows — "
     "more realistic schemas and questions than earlier Spider benchmarks, "
     "but still single-backend (SQL only). Good source for realistic "
     "question-generation patterns to borrow from.",
     ["research", "papers"]),

    ("PolyQuery routing taxonomy",
     "Four routing classes for the benchmark: single-backend (answerable in "
     "one system), multi-backend join (needs results combined across "
     "systems, e.g. Cypher + SQL), code-gen-required (no query language "
     "fits, needs generated Python), and ambiguous (answerable in more than "
     "one place with different cost/quality tradeoffs). The ambiguous class "
     "is where the interesting analysis lives.",
     ["research", "polyquery"]),

    ("Decision log: why hybrid search over pure vector",
     "Chose BM25 + vector hybrid over pure embedding search for vault-mcp "
     "because exact-term queries (like searching for a specific error "
     "message or function name) do much better on keyword match than "
     "semantic similarity. Pure vector search missed exact-string queries "
     "in early testing.",
     ["decisions", "vault-mcp"]),

    ("Decision log: why SQLite over Postgres for vault-mcp",
     "SQLite over Postgres for the note store: this is a single-user local "
     "tool, not a multi-tenant service, so there's no concurrency need "
     "that justifies a client-server database. SQLite's FTS5 extension "
     "gives BM25 for free with zero extra infra.",
     ["decisions", "vault-mcp"]),

    ("Reciprocal Rank Fusion notes",
     "RRF combines rankings from different retrieval methods without "
     "needing their scores to be on the same scale. Formula: for each "
     "item, sum 1/(k + rank) across each ranking it appears in, where k is "
     "a smoothing constant (60 is standard). Used this to combine BM25 and "
     "cosine similarity rankings in vault-mcp.",
     ["research", "vault-mcp"]),

    ("Docker cheat sheet",
     "docker compose up -d — start services in background. docker exec -it "
     "<container> bash — shell into a running container. docker logs -f "
     "<container> — tail logs. docker compose down -v — stop and wipe "
     "volumes (careful, deletes data).",
     ["reference", "docker"]),

    ("FastAPI async patterns",
     "Use async def for route handlers that await I/O (DB calls, HTTP "
     "requests to other services). Don't mark a handler async if it does "
     "blocking CPU work — that blocks the event loop for every other "
     "request. Use run_in_executor or a background worker for CPU-bound "
     "tasks instead.",
     ["reference", "fastapi"]),

    ("Neo4j Cypher syntax notes",
     "Basic pattern: MATCH (a:Person)-[:REFERRED]->(b:Person) WHERE "
     "a.name = 'X' RETURN b. Relationship direction matters — arrows point "
     "from source to target. Use OPTIONAL MATCH for left-join-like "
     "behavior when a relationship might not exist.",
     ["reference", "neo4j", "polyquery"]),

    ("Work log: MCP server week 1",
     "Got the SQLite schema and FTS5 triggers working. Hybrid search with "
     "RRF fusion returns sensible results on manual testing — vector "
     "search correctly surfaces semantically related notes that share no "
     "keywords with the query. Next: build the eval harness before adding "
     "more tools.",
     ["worklog", "vault-mcp"]),

    ("Work log: eval harness",
     "Built recall@k and MRR metrics comparing keyword-only, vector-only, "
     "and hybrid search. Bootstrapped 15 query/relevant-note pairs from my "
     "own vault content. Hybrid should beat both single-method baselines "
     "if the fusion logic is doing its job — that's the headline number "
     "for the README.",
     ["worklog", "vault-mcp"]),

    ("ICONMA internship reframe notes",
     "Reframing the ICONMA internship for SDE-targeted resumes: emphasize "
     "OOP structure of the FastAPI services, REST API design, and the "
     "ML/NLP pipeline as a data pipeline with defined interfaces rather "
     "than a research project.",
     ["interview", "resume"]),

    ("Prompt templating with Jinja2",
     "Jinja2 templates for prompts let you inject schema/context "
     "dynamically without string-formatting spaghetti. Keep templates "
     "versioned separately from application code so prompt changes don't "
     "require a full deploy — this was the basis of the 30% latency cut "
     "from the AutoDashboard templating framework.",
     ["reference", "llm", "strategy"]),

    ("LLM evaluation framework design",
     "Built an eval framework covering 15+ components with automated "
     "checks for accuracy, consistency, and hallucination. Key design "
     "choice: run these checks in CI so a prompt or model change that "
     "regresses quality gets caught before it ships, not after.",
     ["strategy", "llm", "reference"]),

    ("Guardrails vs validation — terminology note",
     "Distinguishing terms I keep mixing up: 'guardrails' = constraints on "
     "what an agent is allowed to do (hard limits, before/during action). "
     "'Validation' = checking whether an output is correct after "
     "generation (can be soft, e.g. a confidence score). Both matter but "
     "solve different failure modes.",
     ["llm", "reference"]),

    ("Fine-tuning vs prompting — when to use which",
     "Rule of thumb from reading around: reach for better prompting/RAG "
     "first — cheaper, faster to iterate, no training infra. Fine-tuning "
     "earns its cost when you need a specific *format* reliably (not just "
     "better knowledge) or need to cut latency/cost by moving to a smaller "
     "model that matches a larger one's behavior on a narrow task.",
     ["llm", "research"]),

    # --- deliberately confusable / adversarial notes below ---
    # These are topically close to earlier notes, which is what makes the
    # eval meaningful: a weak retrieval method can be pulled toward the
    # wrong-but-similar note instead of the exactly right one.

    ("Work log: MCP server week 2",
     "Added the delete_note tool and wired up GitHub Actions CI to run "
     "pytest on every push across Python 3.10-3.12. Also fixed a bug where "
     "updating a note left a stale row in the FTS index.",
     ["worklog", "vault-mcp"]),

    ("FastAPI threadpool gotcha",
     "Calling a blocking library (e.g. a sync DB driver) inside an async "
     "route handler stalls the whole event loop for every concurrent "
     "request, not just that one. Fix: wrap the blocking call with "
     "run_in_threadpool or move it to a background worker.",
     ["reference", "fastapi"]),

    ("PolyQuery ambiguous-class examples",
     "Concrete examples for the ambiguous routing class: 'which customers "
     "referred someone who churned' is answerable via a graph traversal in "
     "Neo4j alone (approximate) or a precise join against the SQL churn "
     "table (exact, slower). The interesting eval question is whether a "
     "model picks the cheap-approximate or correct-expensive path.",
     ["research", "polyquery"]),

    ("SQLite FTS5 sync gotcha",
     "FTS5 doesn't auto-update when the base table changes — you need "
     "explicit triggers on INSERT/UPDATE/DELETE to keep the virtual table "
     "in sync, or search results silently go stale. Learned this the hard "
     "way after an update left duplicate stale rows in the index.",
     ["vault-mcp", "reference"]),

    ("Prompt injection notes",
     "If an agent has a tool that reads untrusted content (web pages, "
     "user-submitted notes, etc), that content could contain instructions "
     "trying to hijack the agent's behavior. Mitigation: treat retrieved "
     "content as data, not instructions, and don't let tool outputs alone "
     "trigger further tool calls without something checking intent.",
     ["llm", "security"]),
]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/tmp/test_vault.db")
    args = parser.parse_args()

    db_path = Path(args.db)
    if db_path.exists():
        db_path.unlink()

    store = VaultStore(db_path)
    print(f"Seeding {len(NOTES)} notes into {db_path} ...")
    for title, content, tags in NOTES:
        note_id = store.add_note(title, content, tags)
        vector = embeddings.embed(f"{title}\n{content}")
        store.set_embedding(note_id, vector)
        print(f"  #{note_id:>2}  {title}")
    print("Done.")


if __name__ == "__main__":
    main()
