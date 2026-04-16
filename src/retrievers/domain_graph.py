"""
ELPIS-Inspired Domain-Partitioned Graph — Approach G3
=======================================================
Divide the corpus by domain, build one HNSW per domain, then bridge
across domains using centroid similarity for cross-domain citations.

Paradigm: Divide and Conquer (Azizi, Echihabi, Palpanas — ELPIS, PVLDB'23).

Key insight: Papers primarily cite within their domain → local graphs first.
Bridge edges via centroid similarity ensure cross-domain citations are kept.

From the paper (page 87): DC methods outperform HNSW on "hard" workloads
(low LID, low LRC). Citation retrieval is hard because citations are sparse
and cross-domain → ELPIS-style partitioning should help.

Expected speedup: 3-8x index build time vs single global graph.

Install: pip install hnswlib
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseRetriever

DATA_DIR = Path(__file__).parent.parent.parent / "data"


class DomainPartitionedGraphRetriever(BaseRetriever):
    """
    ELPIS-inspired divide-and-conquer graph retrieval.

    Algorithm:
    1. Partition corpus by domain (19 natural partitions).
    2. Build local HNSW graph per domain.
    3. Compute domain centroids for medoid-based seed selection.
    4. At query time:
       a. Search the query's own domain graph.
       b. Find the top-K most similar domains by centroid similarity.
       c. Also search those adjacent-domain graphs (bridge connections).
       d. Merge results by cosine similarity score.

    Parameters
    ----------
    n_adjacent_domains : how many extra domains to also search (bridge edges)
    """

    name = "Domain-Partitioned Graph (ELPIS-style)"

    def __init__(
        self,
        emb_col: str = "embedding",
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,
        bridge_k: int = 5,
        n_adjacent_domains: int = 2,
    ):
        self.emb_col = emb_col
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.bridge_k = bridge_k
        self.n_adjacent_domains = n_adjacent_domains
        self.name = (
            f"Domain-Partitioned Graph (M={M}, ef={ef_search}, adj={n_adjacent_domains})"
        )

        self._indices: dict = {}
        self._id_maps: dict[str, list[str]] = {}
        self._domain_centroids: dict[str, np.ndarray] = {}
        self._domain_embs_cache: dict[str, np.ndarray] = {}

        # Pre-computed embeddings support
        self._corpus_embs: np.ndarray | None = None
        self._corpus_ids: list[str] | None = None
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

        self._corpus_embs = corpus_embs
        self._corpus_ids = corpus_ids

        # Group by domain
        domain_groups: dict[str, list[int]] = {}
        for i, (doc_id, dom) in enumerate(zip(corpus_ids, corpus_domains)):
            dom = str(dom) if dom else ""
            if dom and dom not in ("", "nan", "None"):
                domain_groups.setdefault(dom, []).append(i)

        for domain, indices in domain_groups.items():
            if len(indices) < 5:
                continue
            dom_embs = corpus_embs[indices]
            dom_ids = [corpus_ids[i] for i in indices]

            self._indices[domain] = self._make_index(dom_embs)
            self._id_maps[domain] = dom_ids
            self._domain_centroids[domain] = dom_embs.mean(axis=0)
            self._domain_embs_cache[domain] = dom_embs

        print(
            f"  Domain-Partitioned Graph: {len(self._indices)} domain indices "
            f"(M={self.M}, ef={self.ef_search}, adj={self.n_adjacent_domains})"
        )

    def _load_precomputed(self) -> bool:
        emb_dir = DATA_DIR / "embeddings" / "sentence-transformers_all-MiniLM-L6-v2"
        if not emb_dir.exists():
            return False
        try:
            embs = np.load(emb_dir / "corpus_embeddings.npy").astype("float32")
            with open(emb_dir / "corpus_ids.json") as f:
                ids = json.load(f)
            self._query_embs = np.load(emb_dir / "query_embeddings.npy").astype("float32")
            with open(emb_dir / "query_ids.json") as f:
                self._query_ids = json.load(f)
            return True, embs, ids
        except FileNotFoundError:
            return False, None, None

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
        if not self._indices:
            raise RuntimeError("Call fit() before retrieve().")

        use_col = self.emb_col in queries.columns
        id2idx_query = (
            {qid: i for i, qid in enumerate(self._query_ids)}
            if self._query_ids
            else {}
        )

        def _get_q_emb(row) -> np.ndarray:
            if use_col:
                return np.array(row[self.emb_col], dtype="float32").reshape(1, -1)
            return self._query_embs[id2idx_query[row["doc_id"]]].reshape(1, -1)

        results: dict[str, list[str]] = {}

        for _, row in queries.iterrows():
            qid = row["doc_id"]
            q_emb = _get_q_emb(row)
            q_domain = str(row.get("domain", "") or "")

            # ── Select domains to search ─────────────────────────────
            search_domains: list[str] = []
            if q_domain and q_domain in self._indices:
                search_domains.append(q_domain)

            # Find adjacent domains via centroid similarity
            if self._domain_centroids:
                centroid_sims = {
                    d: float(
                        np.dot(q_emb[0], c)
                        / (np.linalg.norm(q_emb[0]) * np.linalg.norm(c) + 1e-9)
                    )
                    for d, c in self._domain_centroids.items()
                    if d not in search_domains
                }
                sorted_doms = sorted(centroid_sims, key=centroid_sims.get, reverse=True)
                search_domains.extend(sorted_doms[: self.n_adjacent_domains])

            # ── Search each relevant domain ──────────────────────────
            all_candidates: dict[str, float] = {}  # doc_id → best sim score

            for dom in search_domains:
                if dom not in self._indices:
                    continue
                dom_ids = self._id_maps[dom]
                k_dom = min(top_k, len(dom_ids))
                labels, distances = self._indices[dom].knn_query(q_emb, k=k_dom)
                for label, dist in zip(labels[0], distances[0]):
                    doc_id = dom_ids[label]
                    if doc_id == qid:
                        continue
                    sim = 1.0 - float(dist)  # cosine similarity from cosine distance
                    if doc_id not in all_candidates or all_candidates[doc_id] < sim:
                        all_candidates[doc_id] = sim

            ranked = sorted(all_candidates, key=all_candidates.get, reverse=True)
            results[qid] = ranked[:top_k]

        return results
