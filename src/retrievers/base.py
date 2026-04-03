"""Abstract base class for all retrievers."""

from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd


class BaseRetriever(ABC):
    """
    All retrievers must implement `fit` and `retrieve`.

    fit(corpus_df)       — index the corpus
    retrieve(queries_df) — return {query_id: [ranked doc_ids, up to 100]}
    """

    name: str = "BaseRetriever"

    @abstractmethod
    def fit(self, corpus: pd.DataFrame) -> None:
        """Index the corpus. Called once before retrieval."""

    @abstractmethod
    def retrieve(self, queries: pd.DataFrame, top_k: int = 100) -> dict[str, list[str]]:
        """
        Return ranked results for every query.

        Parameters
        ----------
        queries : DataFrame with at least columns [doc_id, title, abstract, ta]
        top_k   : number of documents to return per query

        Returns
        -------
        dict mapping query_id -> list of up to top_k corpus doc_ids (ranked)
        """
