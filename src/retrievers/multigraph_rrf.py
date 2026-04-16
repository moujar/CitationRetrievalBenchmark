"""
Multi-Graph RRF Ensemble — Approach G5
========================================
Build separate HNSW proximity graphs per domain AND one global graph,
then fuse their ranked outputs via Reciprocal Rank Fusion (RRF).

Analogous to how Vespa and Elasticsearch 8.x combine HNSW vector search
with other retrieval signals.

RRF formula: score(d) = Σ_i  weight_i / (k_rrf + rank_i(d))

Why this helps:
- Global graph ensures broad semantic coverage
- Domain graph specializes to citation patterns within the same domain
- RRF naturally de-emphasizes documents ranked low in all lists
  and boosts documents that rank high in multiple lists

Reference: Azizi et al. PACMMOD'25, Section VI (multi-graph comparison).

Install: pip install hnswlib
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseRetriever

DATA_DIR = Path(__file__).parent.parent.parent / "data"


class MultiGraphRRF(BaseRetriever):
    """
    Multi-graph RRF ensemble: global HNSW + per-domain HNSW, fused by RRF.

    Parameters
    ----------
    emb_col       : embedding column name in DataFrame (if available)
    rrf_k         : RRF rank-bias constant (default 60, original paper default)
    global_weight : weight for the global graph's ranked list in RRF
    domain_weight : weight for the domain graph's ranked list in RRF
    ef_search     : beam width for HNSW queries
    """

    name = "Multi-Graph RRF"

    def __init__(
        self,
        emb_col: str = "embedding",
        rrf_k: int = 60,
        global_weight: float = 0.5,
        domain_weight: float = 0.5,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,
    ):
        self.emb_col = emb_col
        self.rrf_k = rrf_k
        self.global_weight = global_weight
        self.domain_weight = domain_weight
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.name = (
            f"Multi-Graph RRF (gw={global_weight:.1f}, dw={domain_weight:.1f}, "
            f"k={rrf_k})"
        )

        self._global_index = None
        self._global_id_map: list[str] = []
        self._domain_indices: dict = {}
        self._domain_id_maps: dict[str, list[str]] = {}

        # Pre-computed embeddings support
        self._query_embs: np.ndarray | None = None
        self._query_ids: list[str] | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_index(self, embeddings: np.ndarray):
        import hnswlib
        dim = embeddings.shape[1]
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
            import hnswlib  # noqa: F401
        except ImportError:
            raise ImportError("pip install hnswlib")

        dim = corpus_embs.shape[1]
        self._global_id_map = corpus_ids

        # ── Global index ─────────────────────────────────────────────
        self._global_index = self._make_index(corpus_embs)

        # ── Per-domain indices ────────────────────────────────────────
        domain_groups: dict[str, list[int]] = {}
        for i, (_, dom) in enumerate(zip(corpus_ids, corpus_domains)):
            dom = str(dom) if dom else ""
            if dom and dom not in ("", "nan", "None"):
                domain_groups.setdefault(dom, []).append(i)

        for domain, indices in domain_groups.items():
            if len(indices) < 5:
                continue
            dom_embs = corpus_embs[indices]
            dom_ids = [corpus_ids[i] for i in indices]
            self._domain_indices[domain] = self._make_index(dom_embs)
            self._domain_id_maps[domain] = dom_ids

        print(
            f"  Multi-Graph RRF: 1 global + {len(self._domain_indices)} domain graphs "
            f"(M={self.M}, ef={self.ef_search})"
        )

    def _load_precomputed(self) -> bool:
        emb_dir = DATA_DIR / "embeddings" / "sentence-transformers_all-MiniLM-L6-v2"
        if not emb_dir.exists():
            return False
        try:
            embs = np.load(emb_dir / "corpus_embeddings.npy").astype("float32")
            self._query_embs = np.load(emb_dir / "query_embeddings.npy").astype("float32")
            with open(emb_dir / "corpus_ids.json") as f:
                ids = json.load(f)
            with open(emb_dir / "query_ids.json") as f:
                self._query_ids = json.load(f)
            return True, embs, ids
        except FileNotFoundError:
            return False, None, None

    # ------------------------------------------------------------------
    # RRF merge
    # ------------------------------------------------------------------

    def _rrf_merge(
        self,
        rankings: list[tuple[list[str], float]],
        top_k: int,
        exclude: set[str],
    ) -> list[str]:
        scores: defaultdict[str, float] = defaultdict(float)
        for ranked_list, weight in rankings:
            for rank, doc_id in enumerate(ranked_list, start=1):
                if doc_id not in exclude:
                    scores[doc_id] += weight / (self.rrf_k + rank)
        return sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def fit(self, corpus: pd.DataFrame) -> None:
        domains = corpus["domain"].fillna("").astype(str).tolist()

        if self.emb_col in corpus.columns:
            embs = np.vstack(corpus[self.emb_col].values).astype("float32")
            self._fit_from_arrays(embs, corpus["doc_id"].tolist(), domains)
            return

        result = self._load_precomputed()
        if result[0]:
            _, embs, ids = result
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

        use_col = self.emb_col in queries.columns
        id2idx_q = (
            {qid: i for i, qid in enumerate(self._query_ids)}
            if self._query_ids
            else {}
        )

        def _get_q_emb(row) -> np.ndarray:
            if use_col:
                return np.array(row[self.emb_col], dtype="float32").reshape(1, -1)
            return self._query_embs[id2idx_q[row["doc_id"]]].reshape(1, -1)

        results: dict[str, list[str]] = {}
        fetch_k = min(top_k * 2, len(self._global_id_map))

        for _, row in queries.iterrows():
            qid = row["doc_id"]
            q_emb = _get_q_emb(row)
            q_domain = str(row.get("domain", "") or "")
            exclude = {qid}

            rankings: list[tuple[list[str], float]] = []

            # ── Global ranking ───────────────────────────────────────
            k_gl = min(fetch_k, len(self._global_id_map))
            labels, _ = self._global_index.knn_query(q_emb, k=k_gl)
            global_list = [self._global_id_map[i] for i in labels[0]]
            rankings.append((global_list, self.global_weight))

            # ── Domain ranking ───────────────────────────────────────
            if q_domain and q_domain in self._domain_indices:
                dom_idx = self._domain_indices[q_domain]
                dom_ids = self._domain_id_maps[q_domain]
                k_dom = min(fetch_k, len(dom_ids))
                labels, _ = dom_idx.knn_query(q_emb, k=k_dom)
                dom_list = [dom_ids[i] for i in labels[0]]
                rankings.append((dom_list, self.domain_weight))

            results[qid] = self._rrf_merge(rankings, top_k, exclude)

        return results
