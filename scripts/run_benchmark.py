#!/usr/bin/env python3
"""
IR Challenge — Benchmark Runner
================================
Runs all (or selected) retrievers and prints a leaderboard table.

Usage
-----
# Run all retrievers
python scripts/run_benchmark.py

# Run specific retrievers
python scripts/run_benchmark.py --retrievers tfidf bm25 dense_minilm

# Save results to disk and show per-domain breakdown
python scripts/run_benchmark.py --save --domains

Available retriever keys
------------------------
  tfidf          TF-IDF on title+abstract
  bm25           BM25 on title+abstract
  dense_minilm   Dense (all-MiniLM-L6-v2, pre-computed)
  dense_bge      Dense (BAAI/bge-small-en-v1.5, live-encoded)
  hybrid_bm25    BM25 + Dense MiniLM via RRF
  hybrid_tfidf   TF-IDF + Dense MiniLM via RRF
  rerank_bm25    BM25 → CrossEncoder reranking
  rerank_dense   Dense MiniLM → CrossEncoder reranking
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_queries, load_corpus, load_qrels
from src.evaluate import evaluate, print_results, METRICS
from src.retrievers import (
    TFIDFRetriever,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    CrossEncoderReranker,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# Registry — add new retrievers here
# ---------------------------------------------------------------------------

def build_registry() -> dict:
    dense_minilm = DenseRetriever(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        use_precomputed=True,
    )
    return {
        "tfidf": TFIDFRetriever(field="ta"),
        "bm25": BM25Retriever(field="ta"),
        "dense_minilm": dense_minilm,
        "dense_bge": DenseRetriever(
            model_name="BAAI/bge-small-en-v1.5",
            use_precomputed=True,
        ),
        "hybrid_bm25": HybridRetriever(
            retrievers=[BM25Retriever(field="ta"), DenseRetriever(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                use_precomputed=True,
            )],
        ),
        "hybrid_tfidf": HybridRetriever(
            retrievers=[TFIDFRetriever(field="ta"), DenseRetriever(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                use_precomputed=True,
            )],
        ),
        "rerank_bm25": CrossEncoderReranker(
            first_stage=BM25Retriever(field="ta"),
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            first_stage_k=100,
        ),
        "rerank_dense": CrossEncoderReranker(
            first_stage=DenseRetriever(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                use_precomputed=True,
            ),
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            first_stage_k=100,
        ),
    }


# ---------------------------------------------------------------------------
# Leaderboard printer
# ---------------------------------------------------------------------------

def print_leaderboard(all_scores: list[tuple[str, dict]]) -> None:
    cols = ["NDCG@10", "Recall@10", "Recall@100", "MRR@10", "MAP"]
    col_w = 10
    name_w = 36

    header = f"  {'Retriever':<{name_w}}" + "".join(f"{c:>{col_w}}" for c in cols)
    sep = "  " + "-" * (name_w + col_w * len(cols))

    print(f"\n{'=' * (name_w + col_w * len(cols) + 4)}")
    print("  LEADERBOARD")
    print(f"{'=' * (name_w + col_w * len(cols) + 4)}")
    print(header)
    print(sep)

    ranked = sorted(all_scores, key=lambda x: x[1]["overall"]["NDCG@10"], reverse=True)
    for name, ev in ranked:
        row = f"  {name:<{name_w}}" + "".join(
            f"{ev['overall'].get(c, 0):>{col_w}.4f}" for c in cols
        )
        print(row)
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="IR Challenge benchmark runner")
    parser.add_argument(
        "--retrievers", nargs="+",
        help="Which retrievers to run (default: all). Options: tfidf bm25 dense_minilm dense_bge hybrid_bm25 hybrid_tfidf rerank_bm25 rerank_dense",
    )
    parser.add_argument("--top-k", type=int, default=100, help="Docs to retrieve per query (default 100)")
    parser.add_argument("--save", action="store_true", help="Save ranked results to results/")
    parser.add_argument("--domains", action="store_true", help="Print per-domain breakdown")
    args = parser.parse_args()

    print("Loading data ...")
    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels()

    registry = build_registry()
    keys = args.retrievers if args.retrievers else list(registry.keys())

    invalid = [k for k in keys if k not in registry]
    if invalid:
        print(f"Unknown retrievers: {invalid}. Available: {list(registry.keys())}")
        sys.exit(1)

    RESULTS_DIR.mkdir(exist_ok=True)
    all_scores = []

    for key in keys:
        retriever = registry[key]
        print(f"\n[{key}] {retriever.name}")
        print(f"  Fitting ...")
        t0 = time.time()
        retriever.fit(corpus)
        print(f"  Retrieving ...")
        results = retriever.retrieve(queries, top_k=args.top_k)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

        ev = evaluate(results, qrels, queries_df=queries)
        all_scores.append((retriever.name, ev))
        print_results(retriever.name, ev, show_domains=args.domains)

        if args.save:
            out_path = RESULTS_DIR / f"{key}.json"
            with open(out_path, "w") as f:
                json.dump(results, f)
            print(f"  Saved → {out_path}")

    if len(all_scores) > 1:
        print_leaderboard(all_scores)


if __name__ == "__main__":
    main()
