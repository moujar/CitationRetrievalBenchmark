"""Cross-encoder reranker — wraps any BaseRetriever and reranks its top-k."""

from __future__ import annotations
import pandas as pd
from tqdm import tqdm

from .base import BaseRetriever


class CrossEncoderReranker(BaseRetriever):
    """
    Two-stage retrieval: first-stage retriever + cross-encoder reranking.

    Requires: pip install sentence-transformers

    Parameters
    ----------
    first_stage     : any fitted BaseRetriever (e.g. DenseRetriever, BM25Retriever)
    model_name      : cross-encoder model from HuggingFace
    first_stage_k   : how many candidates to fetch before reranking
    batch_size      : cross-encoder batch size

    Example
    -------
    >>> reranker = CrossEncoderReranker(
    ...     first_stage=BM25Retriever(),
    ...     model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    ... )
    >>> reranker.fit(corpus)
    >>> results = reranker.retrieve(queries, top_k=10)
    """

    name = "CrossEncoder Reranker"

    def __init__(
        self,
        first_stage: BaseRetriever,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        first_stage_k: int = 100,
        batch_size: int = 64,
    ):
        self.first_stage = first_stage
        self.model_name = model_name
        self.first_stage_k = first_stage_k
        self.batch_size = batch_size
        self.name = f"CrossEncoder ({first_stage.name} → {model_name.split('/')[-1]})"
        self._cross_encoder = None
        self._corpus_lookup: dict[str, str] = {}  # doc_id -> text

    def fit(self, corpus: pd.DataFrame) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")

        print(f"  Fitting first stage: {self.first_stage.name} ...")
        self.first_stage.fit(corpus)
        self._corpus_lookup = dict(zip(corpus["doc_id"], corpus["ta"].fillna("")))
        print(f"  Loading cross-encoder: {self.model_name} ...")
        self._cross_encoder = CrossEncoder(self.model_name, max_length=512)

    def retrieve(self, queries: pd.DataFrame, top_k: int = 100) -> dict[str, list[str]]:
        # First stage: get candidates
        candidates = self.first_stage.retrieve(queries, top_k=self.first_stage_k)

        results = {}
        for _, qrow in tqdm(queries.iterrows(), total=len(queries), desc="Reranking", leave=False):
            qid = qrow["doc_id"]
            query_text = qrow["ta"]
            candidate_ids = candidates.get(qid, [])

            if not candidate_ids:
                results[qid] = []
                continue

            pairs = [(query_text, self._corpus_lookup.get(doc_id, "")) for doc_id in candidate_ids]
            scores = self._cross_encoder.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

            ranked = [doc_id for _, doc_id in sorted(zip(scores, candidate_ids), reverse=True)]
            results[qid] = ranked[:top_k]

        return results
