"""
HNSW Dense Retriever — Approach G1
====================================
Drop-in replacement for brute-force cosine similarity using an HNSW index.

Paradigm: Incremental Insertion (Malkov & Yashunin 2016/2018).
Complexity: O(log N) per query vs. O(N) for brute-force.
At 20K docs: index build ~2s, query ~0.1ms — 50-100x faster.

Reference: Azizi et al., PACMMOD'25, Table 1 / Section IV-B.

Install: pip install hnswlib
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseRetriever


class HNSWDenseRetriever(BaseRetriever):
    """
    HNSW-indexed dense retrieval using pre-computed embeddings.

    Hyperparameter recommendations for citation retrieval (20K corpus):
    - M=16          : good balance of speed/recall at 384 dims
    - ef_construction=200 : high-quality graph (increase → better recall)
    - ef_search=100 : beam width at query time (increase → better recall)

    Sweep: M ∈ {8, 16, 32, 64}, ef_search ∈ {50, 100, 200, 400}.
    """

    name = "HNSW Dense"

    def __init__(
        self,
        emb_col: str = "embedding",
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,
        space: str = "cosine",
    ):
        self.emb_col = emb_col
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.space = space
        self.name = f"HNSW Dense (M={M}, ef={ef_search})"

        self._index = None
        self._id_map: list[str] = []
        self._corpus_id_set: set[str] = set()

        # If no stored embeddings column, fall back to pre-computed npy files
        self._query_embs: np.ndarray | None = None
        self._query_ids: list[str] | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_index(self, embeddings: np.ndarray) -> None:
        try:
            import hnswlib
        except ImportError:
            raise ImportError(
                "hnswlib is required for HNSWDenseRetriever.\n"
                "Install with: pip install hnswlib"
            )
        dim = embeddings.shape[1]
        self._index = hnswlib.Index(space=self.space, dim=dim)
        self._index.init_index(
            max_elements=len(embeddings),
            ef_construction=self.ef_construction,
            M=self.M,
        )
        self._index.add_items(embeddings, list(range(len(embeddings))))
        self._index.set_ef(self.ef_search)
        print(
            f"  HNSW index: {len(embeddings)} vectors, "
            f"M={self.M}, ef_construction={self.ef_construction}, "
            f"ef_search={self.ef_search}"
        )

    def _load_precomputed(self) -> bool:
        """Try to load pre-computed embeddings from the standard data directory."""
        from pathlib import Path
        import json

        data_dir = Path(__file__).parent.parent.parent / "data"
        emb_dir = data_dir / "embeddings" / "sentence-transformers_all-MiniLM-L6-v2"
        if not emb_dir.exists():
            return False
        try:
            corpus_embs = np.load(emb_dir / "corpus_embeddings.npy").astype("float32")
            self._query_embs = np.load(emb_dir / "query_embeddings.npy").astype("float32")
            with open(emb_dir / "corpus_ids.json") as f:
                self._id_map = json.load(f)
            with open(emb_dir / "query_ids.json") as f:
                self._query_ids = json.load(f)
            self._corpus_id_set = set(self._id_map)
            self._build_index(corpus_embs)
            print(f"  Loaded pre-computed embeddings from {emb_dir}")
            return True
        except FileNotFoundError:
            return False

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def fit(self, corpus: pd.DataFrame) -> None:
        # Priority 1: use an embedding column already in the DataFrame
        if self.emb_col in corpus.columns:
            embeddings = np.vstack(corpus[self.emb_col].values).astype("float32")
            self._id_map = corpus["doc_id"].tolist()
            self._corpus_id_set = set(self._id_map)
            self._build_index(embeddings)
            return

        # Priority 2: load from pre-computed npy files
        if self._load_precomputed():
            return

        raise ValueError(
            f"Column '{self.emb_col}' not found in corpus and no pre-computed "
            "embeddings available. Run scripts/embed.py first or pass a "
            "DataFrame with an 'embedding' column."
        )

    def retrieve(self, queries: pd.DataFrame, top_k: int = 100) -> dict[str, list[str]]:
        if self._index is None:
            raise RuntimeError("Call fit() before retrieve().")

        results: dict[str, list[str]] = {}
        k = min(top_k, len(self._id_map))

        # ── Use embedding column from queries DataFrame ──────────────
        if self.emb_col in queries.columns:
            for _, row in queries.iterrows():
                qid = row["doc_id"]
                q_emb = np.array(row[self.emb_col], dtype="float32").reshape(1, -1)
                labels, _ = self._index.knn_query(q_emb, k=k)
                ranked = [self._id_map[i] for i in labels[0] if self._id_map[i] != qid]
                results[qid] = ranked[:top_k]
            return results

        # ── Use pre-computed query embeddings ────────────────────────
        if self._query_embs is not None and self._query_ids is not None:
            id2idx = {qid: i for i, qid in enumerate(self._query_ids)}
            for _, row in queries.iterrows():
                qid = row["doc_id"]
                if qid not in id2idx:
                    results[qid] = []
                    continue
                q_emb = self._query_embs[id2idx[qid]].reshape(1, -1)
                labels, _ = self._index.knn_query(q_emb, k=k)
                ranked = [self._id_map[i] for i in labels[0] if self._id_map[i] != qid]
                results[qid] = ranked[:top_k]
            return results

        raise RuntimeError("No query embeddings available. Call fit() first.")
