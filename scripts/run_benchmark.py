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

# List all available retriever keys
python scripts/run_benchmark.py --list

# Run only graph-based retrievers with domain breakdown saved to disk
python scripts/run_benchmark.py --retrievers hnsw_M16 domain_graph multigraph_rrf --save --domains

Available retriever keys (baselines)
-------------------------------------
  tfidf               TF-IDF on title+abstract
  bm25                BM25 on title+abstract
  dense_minilm        Dense (all-MiniLM-L6-v2, pre-computed)
  dense_bge           Dense (BAAI/bge-small-en-v1.5, live-encoded)
  hybrid_bm25         BM25 + Dense MiniLM via RRF
  hybrid_tfidf        TF-IDF + Dense MiniLM via RRF
  rerank_bm25         BM25 -> CrossEncoder reranking
  rerank_dense        Dense MiniLM -> CrossEncoder reranking

Available retriever keys (graph-based -- pip install hnswlib pynndescent)
-------------------------------------------------------------------------
  hnsw_M8/16/32/64       G1: HNSW with varying M (Incremental Insertion)
  hnsw_ef50/100/200/400  G1: HNSW ef_search sweep
  domain_filtered_hnsw   G2: Per-domain HNSW + global fallback (RWalks-style)
  domain_graph           G3: ELPIS-style domain-partitioned graph
  domain_graph_adj3      G3: same + 3 adjacent domains (more bridge edges)
  nndescent              G4: NNDescent direct K-NN
  nndescent_exp1         G4: NNDescent + 1-hop neighborhood expansion
  multigraph_rrf         G5: Multi-graph RRF (global 50% + domain 50%)
  multigraph_rrf_dom_heavy  G5: domain-heavy RRF (30% global, 70% domain)
  hnsw_bm25_rrf          Combined: HNSW + BM25 via RRF (Vespa-style)
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

    registry = {
        # ── Existing baselines ───────────────────────────────────────
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

    # ── Graph-based retrievers (require: pip install hnswlib pynndescent) ──
    try:
        from src.retrievers import (
            HNSWDenseRetriever,
            DomainFilteredHNSW,
            DomainPartitionedGraphRetriever,
            NNDescentGraphExpander,
            MultiGraphRRF,
        )

        graph_registry = {
            # G1: HNSW drop-in — M sweep
            "hnsw_M8":   HNSWDenseRetriever(M=8,  ef_construction=200, ef_search=100),
            "hnsw_M16":  HNSWDenseRetriever(M=16, ef_construction=200, ef_search=100),
            "hnsw_M32":  HNSWDenseRetriever(M=32, ef_construction=200, ef_search=100),
            "hnsw_M64":  HNSWDenseRetriever(M=64, ef_construction=200, ef_search=100),

            # G1: HNSW — ef_search sweep (fixed M=16)
            "hnsw_ef50":  HNSWDenseRetriever(M=16, ef_search=50),
            "hnsw_ef100": HNSWDenseRetriever(M=16, ef_search=100),
            "hnsw_ef200": HNSWDenseRetriever(M=16, ef_search=200),
            "hnsw_ef400": HNSWDenseRetriever(M=16, ef_search=400),

            # G2: Domain-filtered HNSW (RWalks-style predicate search)
            "domain_filtered_hnsw": DomainFilteredHNSW(M=16, ef_search=100),

            # G3: ELPIS-style domain-partitioned graph
            "domain_graph":      DomainPartitionedGraphRetriever(M=16, ef_search=100),
            "domain_graph_adj3": DomainPartitionedGraphRetriever(
                M=16, ef_search=100, n_adjacent_domains=3
            ),

            # G4: NNDescent + graph expansion
            "nndescent":      NNDescentGraphExpander(n_neighbors=30, expansion_hops=0),
            "nndescent_exp1": NNDescentGraphExpander(n_neighbors=30, expansion_hops=1),

            # G5: Multi-graph RRF ensemble
            "multigraph_rrf":           MultiGraphRRF(global_weight=0.5, domain_weight=0.5),
            "multigraph_rrf_dom_heavy": MultiGraphRRF(global_weight=0.3, domain_weight=0.7),

            # Combined: HNSW + BM25 hybrid via RRF (Vespa-style)
            "hnsw_bm25_rrf": HybridRetriever(
                retrievers=[
                    BM25Retriever(field="ta"),
                    HNSWDenseRetriever(M=16, ef_search=100),
                ],
            ),
        }
        registry.update(graph_registry)
        print(f"  [graph] Loaded {len(graph_registry)} graph-based retrievers.")

    except ImportError as e:
        print(
            f"  [graph] Skipping graph-based retrievers — missing dependency: {e}\n"
            "  Install with: pip install hnswlib pynndescent"
        )

    return registry


# ---------------------------------------------------------------------------
# Leaderboard printer
# ---------------------------------------------------------------------------

def print_leaderboard(all_scores: list[tuple[str, dict]]) -> None:
    cols = ["NDCG@10", "Recall@10", "Recall@100", "MRR@10", "MAP"]
    col_w = 10
    name_w = 46

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
        help="Which retrievers to run (default: all). Use --list to see all keys.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available retriever keys and exit.",
    )
    parser.add_argument(
        "--top-k", type=int, default=100,
        help="Docs to retrieve per query (default 100)",
    )
    parser.add_argument("--save", action="store_true", help="Save ranked results to results/")
    parser.add_argument("--domains", action="store_true", help="Print per-domain breakdown")
    args = parser.parse_args()

    registry = build_registry()

    if args.list:
        print("\nAvailable retriever keys:")
        for key in sorted(registry.keys()):
            print(f"  {key:<40}  {registry[key].name}")
        return

    print("Loading data ...")
    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels()

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
