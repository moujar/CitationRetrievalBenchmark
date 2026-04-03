"""Unified evaluation metrics for the IR Challenge benchmark."""

from __future__ import annotations
import math
from collections import defaultdict


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------

def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for doc in ranked[:k] if doc in relevant)
    return hits / len(relevant)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for doc in ranked[:k] if doc in relevant)
    return hits / k


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    for i, doc in enumerate(ranked[:k], start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    def dcg(hits):
        return sum(h / math.log2(i + 2) for i, h in enumerate(hits))

    gains = [1.0 if doc in relevant else 0.0 for doc in ranked[:k]]
    ideal = sorted(gains, reverse=True)
    d = dcg(gains)
    idcg = dcg(ideal)
    return d / idcg if idcg > 0 else 0.0


def average_precision(ranked: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits, ap = 0, 0.0
    for i, doc in enumerate(ranked, start=1):
        if doc in relevant:
            hits += 1
            ap += hits / i
    return ap / len(relevant)


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------

METRICS = ["Recall@10", "Recall@100", "Precision@10", "MRR@10",
           "NDCG@10", "NDCG@100", "MAP"]


def evaluate(
    results: dict[str, list[str]],
    qrels: dict[str, list[str]],
    queries_df=None,
) -> dict:
    """
    Evaluate a submission dict {query_id: [ranked doc_ids]}.

    Returns a dict with:
      - "overall": {metric: value}
      - "per_domain": {domain: {metric: value}}  (only if queries_df provided)
      - "per_query": {query_id: {metric: value}}
    """
    per_query = {}
    domain_scores = defaultdict(lambda: defaultdict(list))

    for qid, ranked in results.items():
        relevant = set(qrels.get(qid, []))
        scores = {
            "Recall@10":    recall_at_k(ranked, relevant, 10),
            "Recall@100":   recall_at_k(ranked, relevant, 100),
            "Precision@10": precision_at_k(ranked, relevant, 10),
            "MRR@10":       mrr_at_k(ranked, relevant, 10),
            "NDCG@10":      ndcg_at_k(ranked, relevant, 10),
            "NDCG@100":     ndcg_at_k(ranked, relevant, 100),
            "MAP":          average_precision(ranked, relevant),
        }
        per_query[qid] = scores

        if queries_df is not None:
            row = queries_df[queries_df["doc_id"] == qid]
            domain = row["domain"].iloc[0] if not row.empty else "unknown"
            for m, v in scores.items():
                domain_scores[domain][m].append(v)

    overall = {
        m: sum(s[m] for s in per_query.values()) / len(per_query)
        for m in METRICS
    }

    per_domain = {
        domain: {m: sum(vals) / len(vals) for m, vals in metrics.items()}
        for domain, metrics in domain_scores.items()
    }

    return {"overall": overall, "per_domain": per_domain, "per_query": per_query}


def print_results(name: str, eval_output: dict, show_domains: bool = False) -> None:
    overall = eval_output["overall"]
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<16} {'Score':>8}")
    print(f"  {'-' * 26}")
    for m in METRICS:
        print(f"  {m:<16} {overall[m]:>8.4f}")

    if show_domains and eval_output.get("per_domain"):
        print(f"\n  Per-domain breakdown:")
        domains = sorted(eval_output["per_domain"].keys())
        header = f"  {'Domain':<30}" + "".join(f" {m:>10}" for m in ["NDCG@10", "Recall@10", "MAP"])
        print(header)
        print("  " + "-" * (len(header) - 2))
        for domain in domains:
            d = eval_output["per_domain"][domain]
            row = f"  {domain:<30}" + "".join(f" {d.get(m, 0):>10.4f}" for m in ["NDCG@10", "Recall@10", "MAP"])
            print(row)
