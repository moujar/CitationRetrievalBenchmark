"""Hybrid retriever combining sparse + dense via Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations
import pandas as pd
from collections import defaultdict

from .base import BaseRetriever


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[str]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    ranked_lists : list of ranked doc-id lists (one per retriever)
    k            : RRF constant (default 60, from original paper)
    weights      : optional per-retriever weights (default: uniform)
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: dict[str, float] = defaultdict(float)
    for ranked, w in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] += w / (k + rank)

    return sorted(scores, key=scores.__getitem__, reverse=True)


class HybridRetriever(BaseRetriever):
    """
    Combines any set of BaseRetriever instances using RRF.

    Parameters
    ----------
    retrievers : list of fitted or unfitted BaseRetriever instances
    weights    : optional per-retriever weights for RRF (default: uniform)
    rrf_k      : RRF constant (default 60)

    Example
    -------
    >>> hybrid = HybridRetriever([BM25Retriever(), DenseRetriever()])
    >>> hybrid.fit(corpus)
    >>> results = hybrid.retrieve(queries)
    """

    name = "Hybrid (RRF)"

    def __init__(
        self,
        retrievers: list[BaseRetriever],
        weights: list[float] | None = None,
        rrf_k: int = 60,
    ):
        if not retrievers:
            raise ValueError("Provide at least one retriever.")
        self.retrievers = retrievers
        self.weights = weights
        self.rrf_k = rrf_k
        names = " + ".join(r.name for r in retrievers)
        self.name = f"Hybrid RRF ({names})"

    def fit(self, corpus: pd.DataFrame) -> None:
        for r in self.retrievers:
            print(f"  Fitting {r.name} ...")
            r.fit(corpus)

    def retrieve(self, queries: pd.DataFrame, top_k: int = 100) -> dict[str, list[str]]:
        # Collect results from each retriever (ask for more than top_k for RRF to work well)
        fetch_k = min(top_k * 3, 1000)
        all_results = [r.retrieve(queries, top_k=fetch_k) for r in self.retrievers]

        fused = {}
        for qid in queries["doc_id"]:
            ranked_lists = [res.get(qid, []) for res in all_results]
            fused[qid] = reciprocal_rank_fusion(ranked_lists, k=self.rrf_k, weights=self.weights)[:top_k]

        return fused
