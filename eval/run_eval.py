"""Compares keyword-only, vector-only, and hybrid search against the
hand-labeled (query -> relevant note) pairs in eval_set.json.

recall@k: of the relevant notes for a query, what fraction show up in the
top k results? MRR: 1 / rank of the first relevant result, averaged across
queries — rewards getting the right answer near the top, not just present.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vault_mcp.search import search  # noqa: E402
from vault_mcp.storage import VaultStore  # noqa: E402

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    if not relevant_ids:
        return 0.0
    hit = len(set(retrieved_ids) & relevant_ids)
    return hit / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    for rank, note_id in enumerate(retrieved_ids, start=1):
        if note_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def run_eval(db_path: str, k: int = 5) -> dict:
    store = VaultStore(db_path)
    eval_set = json.loads(EVAL_SET_PATH.read_text())

    modes = ["keyword", "vector", "hybrid"]
    results = {mode: {"recall": [], "rr": []} for mode in modes}

    per_query_rows = []

    for case in eval_set:
        query = case["query"]
        relevant = set(case["relevant_note_ids"])
        row = {"query": query}

        for mode in modes:
            hits = search(store, query, top_k=k, mode=mode)
            retrieved = [r.note_id for r in hits]
            r_at_k = recall_at_k(retrieved, relevant)
            rr = reciprocal_rank(retrieved, relevant)
            results[mode]["recall"].append(r_at_k)
            results[mode]["rr"].append(rr)
            row[f"{mode}_recall@{k}"] = round(r_at_k, 2)
            row[f"{mode}_rr"] = round(rr, 2)

        per_query_rows.append(row)

    summary = {
        mode: {
            f"recall@{k}": round(sum(vals["recall"]) / len(vals["recall"]), 3),
            "mrr": round(sum(vals["rr"]) / len(vals["rr"]), 3),
        }
        for mode, vals in results.items()
    }

    return {"summary": summary, "per_query": per_query_rows, "n_queries": len(eval_set)}


def print_report(report: dict, k: int) -> None:
    print(f"\nRetrieval eval — {report['n_queries']} queries, k={k}\n")
    header = f"{'mode':<10} {'recall@' + str(k):<12} {'mrr':<8}"
    print(header)
    print("-" * len(header))
    for mode, metrics in report["summary"].items():
        print(f"{mode:<10} {metrics[f'recall@{k}']:<12} {metrics['mrr']:<8}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/tmp/test_vault.db")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="print full JSON report")
    args = parser.parse_args()

    report = run_eval(args.db, k=args.k)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report, args.k)
