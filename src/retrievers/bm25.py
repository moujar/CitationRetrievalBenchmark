"""BM25 retriever using rank_bm25."""

from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm import tqdm

from .base import BaseRetriever


class BM25Retriever(BaseRetriever):
    """
    BM25 sparse retrieval over title + abstract (or full_text).

    Requires: pip install rank-bm25

    Parameters
    ----------
    field : "ta" (title+abstract) | "full_text"
    k1, b  : BM25 hyperparameters
    """

    name = "BM25"

    def __init__(self, field: str = "ta", k1: float = 1.5, b: float = 0.75):
        self.field = field
        self.k1 = k1
        self.b = b
        self._corpus_ids: list[str] = []
        self._bm25 = None

    def fit(self, corpus: pd.DataFrame) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Install rank-bm25: pip install rank-bm25")

        texts = corpus[self.field].fillna("").tolist()
        self._corpus_ids = corpus["doc_id"].tolist()
        tokenized = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)

    def retrieve(self, queries: pd.DataFrame, top_k: int = 100) -> dict[str, list[str]]:
        results = {}
        for _, row in tqdm(queries.iterrows(), total=len(queries), desc="BM25", leave=False):
            qid = row["doc_id"]
            tokens = row[self.field].lower().split()
            scores = self._bm25.get_scores(tokens)
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            ranked = [self._corpus_ids[j] for j in top_indices if self._corpus_ids[j] != qid]
            results[qid] = ranked[:top_k]
        return results
