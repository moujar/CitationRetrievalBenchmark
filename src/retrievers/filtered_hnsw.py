"""
Domain-Filtered HNSW — Approach G2
=====================================
Exploit the 19-domain metadata for filtered vector search.

Inspired by RWalks (AitAomar et al. PACMMOD'25): index-agnostic filtered
search where metadata is diffused along graph edges during construction.

Strategy:
  1. Build a per-domain HNSW index for each of the 19 domains.
  2. Build a global HNSW as fallback for cross-domain queries.
  3. For each query, first retrieve from the query's own domain index,
     then fill remaining slots from global cross-domain search.

This naturally handles the "local domain" recall while preserving the
ability to find cross-domain citations via the global fallback.

Install: pip install hnswlib
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseRetriever

DATA_DIR = Path(__file__).parent.parent.parent / "data"


class DomainFilteredHNSW(BaseRetriever):
    """
    Filtered HNSW: per-domain local indices + global fallback.

    Parameters
    ----------
    emb_col         : column name for embeddings in DataFrame (if available)
    M               : HNSW M parameter (max bidirectional connections per node)
    ef_construction : beam width during index construction
    ef_search       : beam width during search (higher = better recall)
    cross_domain_k  : number of cross-domain candidates from global index
    """

    name = "Domain-Filtered HNSW"

    def __init__(
        self,
        emb_col: str = "embedding",
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,
        cross_domain_k: int = 200,
    ):
        self.emb_col = emb_col
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.cross_domain_k = cross_domain_k
        self.name = f"Domain-Filtered HNSW (M={M}, ef={ef_search})"

        self._global_index = None
        self._global_id_map: list[str] = []
        self._domain_indices: dict = {}
        self._domain_id_maps: dict[str, list[str]] = {}
        self._doc_domains: dict[str, str] = {}

        # Pre-computed embeddings fallback
        self._query_embs: np.ndarray | None = None
        self._query_ids: list[str] | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_index(self, embeddings: np.ndarray, dim: int):
        import hnswlib
        idx = hnswlib.Index(space="cosine", dim=dim)
        idx.init_index(
            max_elements=max(len(embeddings), 10),
            ef_construction=self.ef_construction,
            M=self.M,
        )
        idx.add_items(embeddings, list(range(len(embeddings))))
        idx.set_ef(self.ef_search)
        return idx

    def _fit_from_arrays(
        self,
        corpus_embs: np.ndarray,
        corpus_ids: list[str],
        corpus_domains: list[str],
    ) -> None:
        try:
            import hnswlib  # noqa: F401 — validate import early
        except ImportError:
            raise ImportError("pip install hnswlib")

        dim = corpus_embs.shape[1]
        self._global_id_map = corpus_ids
        self._doc_domains = dict(zip(corpus_ids, corpus_domains))

        # Global index
        self._global_index = self._make_index(corpus_embs, dim)

        # Per-domain indices
        domain_groups: dict[str, list[int]] = {}
        for i, (doc_id, dom) in enumerate(zip(corpus_ids, corpus_domains)):
            if dom and dom not in ("", "nan"):
                domain_groups.setdefault(dom, []).append(i)

        for domain, indices in domain_groups.items():
            if len(indices) < 5:
                continue
            dom_embs = corpus_embs[indices]
            dom_ids = [corpus_ids[i] for i in indices]
            self._domain_indices[domain] = self._make_index(dom_embs, dim)
            self._domain_id_maps[domain] = dom_ids

        print(
            f"  Domain-Filtered HNSW: 1 global + {len(self._domain_indices)} domain indices "
            f"(M={self.M}, ef={self.ef_search})"
        )

    def _load_precomputed(self) -> tuple[np.ndarray, list[str], list[str]] | None:
        emb_dir = DATA_DIR / "embeddings" / "sentence-transformers_all-MiniLM-L6-v2"
        if not emb_dir.exists():
            return None
        try:
            embs = np.load(emb_dir / "corpus_embeddings.npy").astype("float32")
            q_embs = np.load(emb_dir / "query_embeddings.npy").astype("float32")
            with open(emb_dir / "corpus_ids.json") as f:
                ids = json.load(f)
            with open(emb_dir / "query_ids.json") as f:
                self._query_ids = json.load(f)
            self._query_embs = q_embs
            return embs, ids, None  # domains resolved in fit()
        except FileNotFoundError:
            return None

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def fit(self, corpus: pd.DataFrame) -> None:
        domains = corpus["domain"].fillna("").astype(str).tolist()

        if self.emb_col in corpus.columns:
            embs = np.vstack(corpus[self.emb_col].values).astype("float32")
            ids = corpus["doc_id"].tolist()
            self._fit_from_arrays(embs, ids, domains)
            return

        # Pre-computed fallback
        result = self._load_precomputed()
        if result is not None:
            embs, ids, _ = result
            # Align domain order to corpus_ids order from the precomputed file
            id_to_domain = dict(zip(corpus["doc_id"], domains))
            ordered_domains = [id_to_domain.get(i, "") for i in ids]
            self._fit_from_arrays(embs, ids, ordered_domains)
            return

        raise ValueError(
            f"Column '{self.emb_col}' not found and no pre-computed embeddings available."
        )

    def retrieve(self, queries: pd.DataFrame, top_k: int = 100) -> dict[str, list[str]]:
        if self._global_index is None:
            raise RuntimeError("Call fit() before retrieve().")

        # Resolve query embeddings
        use_col = self.emb_col in queries.columns

        def _get_q_emb(row) -> np.ndarray:
            if use_col:
                return np.array(row[self.emb_col], dtype="float32").reshape(1, -1)
            id2idx = {qid: i for i, qid in enumerate(self._query_ids)}
            return self._query_embs[id2idx[row["doc_id"]]].reshape(1, -1)

        results: dict[str, list[str]] = {}

        for _, row in queries.iterrows():
            qid = row["doc_id"]
            q_emb = _get_q_emb(row)
            q_domain = str(row.get("domain", "") or "")

            candidates: list[str] = []
            seen: set[str] = {qid}  # exclude query itself

            # ── 1. Domain-local search ───────────────────────────────
            if q_domain and q_domain in self._domain_indices:
                dom_idx = self._domain_indices[q_domain]
                dom_ids = self._domain_id_maps[q_domain]
                k_dom = min(top_k, len(dom_ids))
                labels, _ = dom_idx.knn_query(q_emb, k=k_dom)
                for i in labels[0]:
                    doc_id = dom_ids[i]
                    if doc_id not in seen:
                        candidates.append(doc_id)
                        seen.add(doc_id)

            # ── 2. Global cross-domain fill ──────────────────────────
            remaining = top_k - len(candidates)
            if remaining > 0:
                k_global = min(self.cross_domain_k, len(self._global_id_map))
                labels, _ = self._global_index.knn_query(q_emb, k=k_global)
                for i in labels[0]:
                    doc_id = self._global_id_map[i]
                    if doc_id not in seen:
                        candidates.append(doc_id)
                        seen.add(doc_id)
                    if len(candidates) >= top_k:
                        break

            results[qid] = candidates[:top_k]

        return results
