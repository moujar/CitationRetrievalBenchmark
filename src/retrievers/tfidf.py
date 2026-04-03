"""TF-IDF sparse retriever."""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseRetriever


class TFIDFRetriever(BaseRetriever):
    """
    Sparse TF-IDF retrieval over title + abstract.

    Parameters
    ----------
    field : "ta" (title+abstract) | "full_text"
    """

    name = "TF-IDF"

    def __init__(self, field: str = "ta"):
        self.field = field
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 1),
            stop_words="english",
        )
        self._corpus_ids: list[str] = []
        self._corpus_matrix = None

    def fit(self, corpus: pd.DataFrame) -> None:
        texts = corpus[self.field].fillna("").tolist()
        self._corpus_ids = corpus["doc_id"].tolist()
        self._corpus_matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, queries: pd.DataFrame, top_k: int = 100) -> dict[str, list[str]]:
        results = {}
        query_texts = queries[self.field].fillna("").tolist()
        query_matrix = self.vectorizer.transform(query_texts)

        for i, qid in enumerate(queries["doc_id"]):
            scores = cosine_similarity(query_matrix[i], self._corpus_matrix).flatten()
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            ranked = [self._corpus_ids[j] for j in top_indices if self._corpus_ids[j] != qid]
            results[qid] = ranked[:top_k]

        return results
