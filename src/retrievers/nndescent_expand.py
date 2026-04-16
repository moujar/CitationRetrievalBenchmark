"""
NNDescent Graph + Neighborhood Expansion — Approach G4
========================================================
Build a K-NN proximity graph using NNDescent (PyNNDescent), then retrieve
by (1) direct K-NN lookup and (2) optional 1-hop neighborhood expansion.

Paradigm: Neighborhood Propagation / KGraph (Dong et al. WWW 2011).

The 1-hop expansion mimics citation graph traversal:
"If paper A is a near-neighbor of query Q, then A's neighbors are likely
relevant to Q as well" — analogous to citation diffusion.

This is especially powerful for sparse cross-domain citations that may be
missed by direct K-NN (low LRC datasets — see PDF pages 42-43).

Install: pip install pynndescent
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseRetriever

DATA_DIR = Path(__file__).parent.parent.parent / "data"


class NNDescentGraphExpander(BaseRetriever):
    """
    NNDescent K-NN graph construction + neighborhood expansion retrieval.

    Parameters
    ----------
    emb_col         : embedding column name (if DataFrame has embeddings)
    n_neighbors     : K-NN graph degree (30 recommended for citation retrieval)
    expansion_hops  : 0 = direct K-NN only; 1 = 1-hop expansion (recommended)
    """

    name = "NNDescent Graph + Expansion"

    def __init__(
        self,
        emb_col: str = "embedding",
        n_neighbors: int = 30,
        expansion_hops: int = 1,
    ):
        self.emb_col = emb_col
        self.n_neighbors = n_neighbors
        self.expansion_hops = expansion_hops
        self.name = f"NNDescent (k={n_neighbors}, hops={expansion_hops})"

        self._index = None
        self._raw_embs: np.ndarray | None = None
        self._id_map: list[str] = []

        # Pre-computed embeddings support
        self._query_embs: np.ndarray | None = None
        self._query_ids: list[str] | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_from_arrays(self, corpus_embs: np.ndarray, corpus_ids: list[str]) -> None:
        try:
            import pynndescent
        except ImportError:
            raise ImportError(
                "pynndescent is required for NNDescentGraphExpander.\n"
                "Install with: pip install pynndescent"
            )
        self._raw_embs = corpus_embs
        self._id_map = corpus_ids
        print(f"  Building NNDescent graph on {len(corpus_embs)} vectors "
              f"(k={self.n_neighbors}) ...")
        self._index = pynndescent.NNDescent(
            corpus_embs,
            n_neighbors=self.n_neighbors,
            metric="cosine",
            n_jobs=-1,
            verbose=False,
        )
        self._index.prepare()
        print("  NNDescent graph ready.")

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
            self._fit_from_arrays(embs, ids)
            return True
        except FileNotFoundError:
            return False

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def fit(self, corpus: pd.DataFrame) -> None:
        if self.emb_col in corpus.columns:
            embs = np.vstack(corpus[self.emb_col].values).astype("float32")
            self._fit_from_arrays(embs, corpus["doc_id"].tolist())
            return

        if not self._load_precomputed():
            raise ValueError(
                f"Column '{self.emb_col}' not found and no pre-computed embeddings available."
            )

    def retrieve(self, queries: pd.DataFrame, top_k: int = 100) -> dict[str, list[str]]:
        if self._index is None:
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
        n = len(self._id_map)

        for _, row in queries.iterrows():
            qid = row["doc_id"]
            q_emb = _get_q_emb(row)

            # ── Step 1: Direct K-NN lookup ───────────────────────────
            k_init = min(top_k, n)
            nn_indices, nn_dists = self._index.query(q_emb, k=k_init)
            nn_indices = nn_indices[0]
            direct_ids = [self._id_map[i] for i in nn_indices if self._id_map[i] != qid]

            if self.expansion_hops <= 0:
                results[qid] = direct_ids[:top_k]
                continue

            # ── Step 2: 1-hop neighborhood expansion ────────────────
            expanded_set: set[str] = set(direct_ids)
            expanded_set.discard(qid)
            frontier = list(nn_indices[: self.n_neighbors])

            for _hop in range(self.expansion_hops):
                new_frontier = []
                for node_idx in frontier:
                    # Get pre-built neighbors in the graph
                    graph_neighbors = self._index.neighbor_graph[0][node_idx]
                    for hi in graph_neighbors:
                        if hi < 0 or hi >= n:
                            continue
                        hop_id = self._id_map[hi]
                        if hop_id != qid and hop_id not in expanded_set:
                            expanded_set.add(hop_id)
                            new_frontier.append(hi)
                frontier = new_frontier[: self.n_neighbors]

            # ── Step 3: Re-rank expanded set by cosine similarity ────
            all_ids = list(expanded_set)
            if not all_ids:
                results[qid] = direct_ids[:top_k]
                continue

            id_to_idx = {did: i for i, did in enumerate(self._id_map)}
            valid_ids = [d for d in all_ids if d in id_to_idx]
            if not valid_ids:
                results[qid] = direct_ids[:top_k]
                continue

            valid_embs = np.vstack(
                [self._raw_embs[id_to_idx[d]] for d in valid_ids]
            ).astype("float32")
            sims = (valid_embs @ q_emb.T).squeeze()
            ranked_idx = np.argsort(-sims)
            ranked_ids = [valid_ids[i] for i in ranked_idx]

            results[qid] = ranked_ids[:top_k]

        return results
