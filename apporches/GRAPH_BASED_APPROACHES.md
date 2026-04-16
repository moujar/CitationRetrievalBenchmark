# Graph-Based Vector Search for Scientific Citation Retrieval
## Merging Proximity Graph Methods with the IR Challenge

> Based on: *Graph-Based Vector Search* — Prof. Karima Echihabi, UM6P / University of Waterloo, 2025  
> Challenge: Citation-based scientific paper retrieval from 20K-paper corpus (Semantic Scholar)  
> Task: Given a query paper, retrieve the papers it cites. Corpus = 20,000 papers, 100 queries, 19 domains.

---

## Why Graph-Based Search Fits This Challenge

The citation retrieval task has specific properties that make graph-based ANNS ideal:

| Property | Challenge Value | Graph Implication |
|---|---|---|
| Corpus size | 20,000 docs | Small-medium dataset → NSG, HNSW, or ELPIS all viable |
| Embedding dim | 384 (all-MiniLM-L6-v2) | Moderate dim → graph traversal efficient |
| Query type | Top-K retrieval (K=100) | Beam search naturally outputs ranked lists |
| Metadata | domain (19), year, venue | Supports filtered/predicate search (RWalks-style) |
| Citation structure | Known ground truth citations | Can build a *citation proximity graph* as auxiliary index |

**Current baseline**: Brute-force cosine similarity (O(N) per query).  
**With HNSW/NSG**: O(log N) per query at 0.99 recall — same quality, 10-100x faster.  
**Key insight**: The 20K corpus has *structure* (19 domains, citation graph) that graph-based methods can exploit, whereas brute-force ignores it entirely.

---

## The 5-Paradigm Taxonomy Applied to Citation Retrieval

From Azizi et al. PACMMOD'25, graph-based ANNS methods fall into 5 design paradigms. Here is how each maps to the challenge:

### Paradigm 1 — Neighborhood Propagation (KGraph, EFANNA, IEH)
**Core idea**: "Your neighbors' neighbors are likely my neighbors" (NNDescent principle).  
**Application**: Build a K-NN graph over all 20K paper embeddings using NNDescent. Use this graph for both retrieval AND query expansion.

```
Algorithm NNDescent-Citation:
1. Initialize: assign random K neighbors to each paper
2. For each paper p:
   - Get neighbors N(p)
   - For each neighbor n ∈ N(p):
       - Candidates = N(n)  ← neighbors' neighbors
       - Update N(p) if any candidate is closer to p
3. Repeat until convergence
```

**Research contribution**: Use the converged K-NN graph as a *citation proxy* — if paper A and B are close in embedding space, and B cites C, then C is likely relevant to A even without direct citation overlap. This is **graph-based citation diffusion**.

---

### Paradigm 2 — Incremental Insertion (NSW → HNSW, Vamana)
**Core idea**: Insert papers one-by-one into a navigable proximity graph using beam search.  
**Recommended for this challenge**: HNSW (most widely deployed, excellent recall/speed tradeoff at 20K scale).

**HNSW on 20K papers:**
```python
import hnswlib
import numpy as np

# Build HNSW index
dim = 384
index = hnswlib.Index(space='cosine', dim=dim)
index.init_index(max_elements=20000, ef_construction=200, M=16)
index.add_items(corpus_embeddings, corpus_ids)
index.set_ef(50)  # ef > top_k for better recall

# Query
labels, distances = index.knn_query(query_embedding, k=100)
```

**Key hyperparameters** (tuned for citation retrieval):
- `M=16`: each node has max 16 bidirectional neighbors → good for 384-dim
- `ef_construction=200`: beam width during indexing → higher = better graph quality
- `ef=50..200`: beam width during search → trade speed for recall
- At 20K docs: indexing takes ~2 seconds, search ~0.1ms per query

**Experimental result (from PDF, page 86)**: On Deep1M and ImageNet1M (comparable to 20K at inference scale), ND-based methods (HNSW uses RRND via α-relaxed RNG) lead best search performance.

---

### Paradigm 3 — Neighborhood Diversification (NSG, SSG, DPG)
**Core idea**: Prune graph edges to ensure diverse neighborhood coverage, approximating the Relative Neighborhood Graph (RNG).

**Three ND variants and their trade-offs for citation retrieval:**

| Variant | Pruning Rate | Search Efficiency | Best for |
|---|---|---|---|
| **MOND** (angle-based) | <5% | Moderate | When preserving clusters |
| **RND** (relative ND) | 5-20% | **Best overall** | Citation retrieval (diverse domains) |
| **RRND** (relaxed RND, α>1) | <1% | Good | When recall is paramount |

**Recommendation**: Use RND-based pruning (used by NSG). RND prunes 5-20x more than MOND → smaller index → faster traversal. From the PDF (page 82): *"RND leads to the best search efficiency across datasets and dataset sizes."*

**Why RND helps citation retrieval**: Papers cite across domains (a CS paper might cite a statistics paper). RND ensures neighbors span diverse semantic regions, not just the local domain cluster → better cross-domain recall.

**Implementation via NSG (Navigating Spreading-out Graph):**
```python
# Install: pip install pynndescent (for base K-NN) + custom NSG or use nmslib
import nmslib

index = nmslib.init(method='nsg', space='cosinesimil')
index.addDataPointBatch(corpus_embeddings)
index.createIndex({'post': 2, 'R': 32, 'L': 100, 'C': 500})
index.setQueryTimeParams({'efSearch': 100})

results = index.knnQueryBatch(query_embeddings, k=100, num_threads=4)
```

---

### Paradigm 4 — Divide and Conquer (SPTAG, HCNNG, ELPIS)
**Core idea**: Recursively partition the dataset, build local proximity graphs per partition, connect them with inter-partition edges.

**This paradigm is most novel for citation retrieval** because the challenge has natural partitions: **19 domains**.

**ELPIS-inspired domain-partitioned graph (new research contribution):**

```
Algorithm CitationELPIS:
1. Partition corpus by domain → 19 domain clusters D_1, ..., D_19
   (average ~1050 papers per domain)
   
2. For each domain D_i:
   Build local HNSW/NSG graph G_i on papers in D_i
   
3. Build a "bridge graph":
   For each domain D_i, find the k=5 nearest papers in each other domain D_j
   Add these as inter-partition edges
   
4. Query(q):
   a. Classify q to domain d* (use query's domain field)
   b. Enter G_{d*} at the medoid of d*
   c. Beam search within G_{d*} for top-K local results
   d. Optionally: follow bridge edges to adjacent domains
   e. Return merged ranked list
```

**Why this outperforms flat HNSW for citation retrieval**:
- Papers primarily cite within their domain → local graph quality matters most
- Smaller per-domain graphs (1050 nodes) → faster beam search
- Bridge edges ensure cross-domain citations are captured
- Matches ELPIS principle: DC methods lead best on "hard" workloads (page 87); citation retrieval is hard because citations are sparse and cross-domain

**Expected speedup**: 3-8x index build time vs. global graph (from PDF, page 65: ELPIS builds 3-8x faster than SOTA).

---

### Paradigm 5 — Hierarchy / Multi-Layer (HNSW multi-layer, Vamana/DiskANN)
**Core idea**: Build a hierarchy of graphs — upper layers are sparse long-range connections, lower layers are dense fine-grained connections.

**Application to citation retrieval:**

HNSW's hierarchy maps naturally to a **coarse-to-fine semantic hierarchy** of papers:
- **Layer 2 (top)**: ~50 papers — "field representatives" (one per domain + subdomain)
- **Layer 1 (middle)**: ~1000 papers — venue/year cluster representatives
- **Layer 0 (bottom)**: all 20,000 papers — full dense graph

**Seed selection with medoid (from PDF, page 83 — MD seed strategy):**
```python
def domain_medoid_entry(query_domain, domain_centroids):
    """Use the medoid of the query's domain cluster as HNSW entry point."""
    entry_id = domain_centroids[query_domain]  # precomputed
    return entry_id

# In HNSW search, override the entry node to the domain medoid
# instead of the global medoid → shorter path to the target region
```

**Experimental finding (page 83)**: Medoid (MD) seed selection is one of the best strategies for indexing performance. For this challenge, using the query paper's *domain medoid* as the HNSW entry point improves both speed and recall.

---

## Concrete New Retriever Implementations

### Approach G1: HNSW Dense Retriever (Drop-in Replacement)

Replace brute-force cosine similarity with HNSW-indexed search. Same recall, O(log N) query time.

```python
# src/retrievers/hnsw_dense.py
from __future__ import annotations
import numpy as np
import pandas as pd
import hnswlib
from .base import BaseRetriever


class HNSWDenseRetriever(BaseRetriever):
    """
    HNSW-indexed dense retrieval.
    Incremental Insertion paradigm (Malkov & Yashunin 2016).
    ~50-100x faster than brute-force at same recall level.
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
        self._index = None
        self._id_map: list[str] = []

    def fit(self, corpus: pd.DataFrame) -> None:
        embeddings = np.vstack(corpus[self.emb_col].values).astype("float32")
        self._id_map = corpus["doc_id"].tolist()
        dim = embeddings.shape[1]

        self._index = hnswlib.Index(space=self.space, dim=dim)
        self._index.init_index(
            max_elements=len(embeddings),
            ef_construction=self.ef_construction,
            M=self.M,
        )
        self._index.add_items(embeddings, list(range(len(embeddings))))
        self._index.set_ef(self.ef_search)
        print(f"  HNSW index built: {len(embeddings)} vectors, M={self.M}, ef_c={self.ef_construction}")

    def retrieve(self, queries: pd.DataFrame, top_k: int = 100) -> dict[str, list[str]]:
        results = {}
        for _, row in queries.iterrows():
            qid = row["doc_id"]
            q_emb = np.array(row[self.emb_col], dtype="float32").reshape(1, -1)
            labels, _ = self._index.knn_query(q_emb, k=min(top_k, len(self._id_map)))
            results[qid] = [self._id_map[i] for i in labels[0]]
        return results
```

**Hyperparameter sweep for benchmark paper:**
```
M ∈ {8, 16, 32, 64}
ef_construction ∈ {100, 200, 400}
ef_search ∈ {50, 100, 200, 400}
```

---

### Approach G2: Domain-Filtered HNSW (RWalks-style Predicate Search)

Exploit the 19-domain metadata for filtered vector search. When a query paper belongs to domain D, restrict candidates to papers in domain D (and neighbors).

```python
# src/retrievers/filtered_hnsw.py
from __future__ import annotations
import numpy as np
import pandas as pd
import hnswlib
from .base import BaseRetriever


class DomainFilteredHNSW(BaseRetriever):
    """
    Filtered HNSW: restrict search to query's domain + cross-domain neighbors.
    Inspired by RWalks (AitAomar et al. PACMMOD'25): index-agnostic filtered search
    where metadata is diffused along random walks during graph construction.
    """
    name = "Domain-Filtered HNSW"

    def __init__(
        self,
        emb_col: str = "embedding",
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,
        cross_domain_k: int = 200,   # bridge candidates from other domains
        use_full_corpus_fallback: bool = True,
    ):
        self.emb_col = emb_col
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.cross_domain_k = cross_domain_k
        self.use_full_corpus_fallback = use_full_corpus_fallback
        self._global_index = None
        self._domain_indices: dict[str, hnswlib.Index] = {}
        self._domain_id_maps: dict[str, list[str]] = {}
        self._global_id_map: list[str] = []
        self._doc_domains: dict[str, str] = {}

    def fit(self, corpus: pd.DataFrame) -> None:
        embeddings = np.vstack(corpus[self.emb_col].values).astype("float32")
        dim = embeddings.shape[1]
        self._global_id_map = corpus["doc_id"].tolist()
        self._doc_domains = dict(zip(corpus["doc_id"], corpus["domain"].fillna("unknown")))

        # Build global HNSW fallback
        self._global_index = hnswlib.Index(space="cosine", dim=dim)
        self._global_index.init_index(max_elements=len(embeddings), ef_construction=self.ef_construction, M=self.M)
        self._global_index.add_items(embeddings, list(range(len(embeddings))))
        self._global_index.set_ef(self.ef_search)

        # Build per-domain HNSW indices
        for domain, group in corpus.groupby("domain"):
            if pd.isna(domain):
                continue
            dom_embs = np.vstack(group[self.emb_col].values).astype("float32")
            dom_ids = group["doc_id"].tolist()
            if len(dom_ids) < 10:
                continue
            idx = hnswlib.Index(space="cosine", dim=dim)
            idx.init_index(max_elements=len(dom_ids), ef_construction=self.ef_construction, M=self.M)
            idx.add_items(dom_embs, list(range(len(dom_ids))))
            idx.set_ef(self.ef_search)
            self._domain_indices[domain] = idx
            self._domain_id_maps[domain] = dom_ids
        print(f"  Built {len(self._domain_indices)} domain HNSW indices")

    def retrieve(self, queries: pd.DataFrame, top_k: int = 100) -> dict[str, list[str]]:
        results = {}
        for _, row in queries.iterrows():
            qid = row["doc_id"]
            q_emb = np.array(row[self.emb_col], dtype="float32").reshape(1, -1)
            q_domain = row.get("domain", None)

            candidates = []

            # Domain-filtered search
            if q_domain and q_domain in self._domain_indices:
                dom_idx = self._domain_indices[q_domain]
                dom_ids = self._domain_id_maps[q_domain]
                k_dom = min(top_k, len(dom_ids))
                labels, _ = dom_idx.knn_query(q_emb, k=k_dom)
                candidates.extend([dom_ids[i] for i in labels[0]])

            # Cross-domain bridge: fill remaining slots from global index
            remaining = top_k - len(candidates)
            if remaining > 0 and self._global_index:
                k_global = min(self.cross_domain_k, len(self._global_id_map))
                labels, _ = self._global_index.knn_query(q_emb, k=k_global)
                global_cands = [self._global_id_map[i] for i in labels[0]]
                # Add cross-domain candidates not already included
                for doc_id in global_cands:
                    if doc_id not in set(candidates):
                        candidates.append(doc_id)
                    if len(candidates) >= top_k:
                        break

            results[qid] = candidates[:top_k]
        return results
```

---

### Approach G3: ELPIS-Inspired Domain-Partitioned Graph

Build one graph per domain, connect them via inter-domain bridges, use domain knowledge for guided entry.

```python
# src/retrievers/domain_graph.py
from __future__ import annotations
import numpy as np
import pandas as pd
import hnswlib
from .base import BaseRetriever


class DomainPartitionedGraphRetriever(BaseRetriever):
    """
    ELPIS-inspired divide-and-conquer graph retrieval.
    
    Partition the corpus by domain, build local proximity graphs,
    then bridge across domains via inter-partition connections.
    
    Reference: Azizi, Echihabi, Palpanas. ELPIS: Graph-Based Similarity Search
    for Scalable Data Science. PVLDB 16(6): 1548-1559 (2023).
    """
    name = "Domain-Partitioned Graph (ELPIS-style)"

    def __init__(
        self,
        emb_col: str = "embedding",
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,
        bridge_k: int = 5,          # top-K inter-domain bridges per domain
    ):
        self.emb_col = emb_col
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.bridge_k = bridge_k
        self._indices: dict[str, hnswlib.Index] = {}
        self._id_maps: dict[str, list[str]] = {}
        self._domain_centroids: dict[str, np.ndarray] = {}
        self._domain_order: list[str] = []

    def fit(self, corpus: pd.DataFrame) -> None:
        dim = len(corpus[self.emb_col].iloc[0])

        for domain, group in corpus.groupby("domain"):
            if pd.isna(domain) or len(group) < 5:
                continue
            dom_embs = np.vstack(group[self.emb_col].values).astype("float32")
            dom_ids = group["doc_id"].tolist()

            # Build local domain graph
            idx = hnswlib.Index(space="cosine", dim=dim)
            idx.init_index(max_elements=len(dom_ids), ef_construction=self.ef_construction, M=self.M)
            idx.add_items(dom_embs, list(range(len(dom_ids))))
            idx.set_ef(self.ef_search)
            self._indices[domain] = idx
            self._id_maps[domain] = dom_ids

            # Compute domain centroid (for medoid-based seed selection)
            self._domain_centroids[domain] = dom_embs.mean(axis=0)

        self._domain_order = list(self._indices.keys())
        print(f"  Built {len(self._indices)} domain graphs")

    def retrieve(self, queries: pd.DataFrame, top_k: int = 100) -> dict[str, list[str]]:
        results = {}
        for _, row in queries.iterrows():
            qid = row["doc_id"]
            q_emb = np.array(row[self.emb_col], dtype="float32").reshape(1, -1)
            q_domain = row.get("domain", None)

            all_candidates = {}  # doc_id -> score (1 - cosine_dist)

            # Primary domain search
            primary_domains = []
            if q_domain and q_domain in self._indices:
                primary_domains.append(q_domain)

            # Find adjacent domains by centroid similarity
            if self._domain_centroids:
                centroid_sims = {
                    d: float(np.dot(q_emb[0], c) / (np.linalg.norm(q_emb[0]) * np.linalg.norm(c) + 1e-9))
                    for d, c in self._domain_centroids.items()
                }
                sorted_domains = sorted(centroid_sims, key=centroid_sims.get, reverse=True)
                # Search top-3 most similar domains (bridge connections)
                for dom in sorted_domains[:3]:
                    if dom not in primary_domains:
                        primary_domains.append(dom)

            # Search each relevant domain
            for dom in primary_domains:
                if dom not in self._indices:
                    continue
                k_dom = min(top_k, len(self._id_maps[dom]))
                labels, distances = self._indices[dom].knn_query(q_emb, k=k_dom)
                for label, dist in zip(labels[0], distances[0]):
                    doc_id = self._id_maps[dom][label]
                    score = 1.0 - dist  # cosine similarity
                    if doc_id not in all_candidates or all_candidates[doc_id] < score:
                        all_candidates[doc_id] = score

            ranked = sorted(all_candidates, key=all_candidates.get, reverse=True)
            results[qid] = ranked[:top_k]
        return results
```

---

### Approach G4: NNDescent Citation Graph + Graph Expansion

Build a K-NN graph using NNDescent, then use it for query expansion by walking the citation-proxy graph.

```python
# src/retrievers/nndescent_expand.py
from __future__ import annotations
import numpy as np
import pandas as pd
from .base import BaseRetriever


class NNDescentGraphExpander(BaseRetriever):
    """
    NNDescent K-NN graph construction + neighborhood expansion retrieval.
    
    Builds a K-NN proximity graph using PyNNDescent (NNDescent algorithm),
    then retrieves by: (1) direct K-NN lookup, (2) 1-hop neighborhood expansion.
    
    The 1-hop expansion mimics how citation graphs work: if paper A is a
    near-neighbor of query Q, then A's neighbors may also be relevant to Q.
    
    Reference: Dong et al. NNDescent (KGraph), 2011.
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
        self._index = None
        self._id_map: list[str] = []

    def fit(self, corpus: pd.DataFrame) -> None:
        try:
            import pynndescent
        except ImportError:
            raise ImportError("pip install pynndescent")

        embeddings = np.vstack(corpus[self.emb_col].values).astype("float32")
        self._id_map = corpus["doc_id"].tolist()
        print(f"  Building NNDescent graph on {len(embeddings)} vectors ...")
        self._index = pynndescent.NNDescent(
            embeddings,
            n_neighbors=self.n_neighbors,
            metric="cosine",
            n_jobs=-1,
            verbose=True,
        )
        self._index.prepare()
        print("  NNDescent graph ready.")

    def retrieve(self, queries: pd.DataFrame, top_k: int = 100) -> dict[str, list[str]]:
        results = {}
        for _, row in queries.iterrows():
            qid = row["doc_id"]
            q_emb = np.array(row[self.emb_col], dtype="float32").reshape(1, -1)

            # Step 1: Direct K-NN lookup
            nn_indices, nn_dists = self._index.query(q_emb, k=min(top_k, len(self._id_map)))
            direct_ids = [self._id_map[i] for i in nn_indices[0]]

            # Step 2: 1-hop neighborhood expansion
            if self.expansion_hops > 0:
                expanded = set(direct_ids)
                frontier = list(nn_indices[0][:self.n_neighbors])
                for _ in range(self.expansion_hops):
                    next_frontier = []
                    for node_idx in frontier:
                        hop_indices, _ = self._index.query(
                            self._index._raw_data[node_idx:node_idx+1], k=self.n_neighbors
                        )
                        for hi in hop_indices[0]:
                            hop_id = self._id_map[hi]
                            if hop_id not in expanded:
                                expanded.add(hop_id)
                                next_frontier.append(hi)
                    frontier = next_frontier[:self.n_neighbors]

                # Re-rank expanded set by cosine similarity to query
                all_ids = list(expanded)
                all_embs = np.vstack([
                    self._index._raw_data[self._id_map.index(d)]
                    for d in all_ids
                    if d in self._id_map
                ]).astype("float32")
                sims = all_embs @ q_emb.T
                ranked_idx = np.argsort(-sims.squeeze())
                direct_ids = [all_ids[i] for i in ranked_idx[:top_k]]

            results[qid] = direct_ids[:top_k]
        return results
```

---

### Approach G5: Multi-Graph Ensemble (Per-Domain + Global, fused via RRF)

Build separate proximity graphs per domain AND a global graph. Fuse their ranked outputs via RRF — analogous to how HNSW + BM25 is used in Vespa.

```python
# src/retrievers/multigraph_rrf.py
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd
import hnswlib
from .base import BaseRetriever


class MultiGraphRRF(BaseRetriever):
    """
    Multi-graph ensemble with Reciprocal Rank Fusion.
    
    Builds:
    - Global HNSW over all 20K papers
    - Per-domain HNSW over each of the 19 domains
    
    Fuses rankings via RRF: score(d) = Σ_i w_i / (k + rank_i(d))
    
    This is directly analogous to how Vespa and Elasticsearch 8.x combine
    HNSW vector search with other retrieval signals.
    """
    name = "Multi-Graph RRF"

    def __init__(
        self,
        emb_col: str = "embedding",
        rrf_k: int = 60,
        global_weight: float = 0.5,
        domain_weight: float = 0.5,
        ef_search: int = 100,
    ):
        self.emb_col = emb_col
        self.rrf_k = rrf_k
        self.global_weight = global_weight
        self.domain_weight = domain_weight
        self.ef_search = ef_search
        self._global_index = None
        self._global_id_map: list[str] = []
        self._domain_indices: dict[str, hnswlib.Index] = {}
        self._domain_id_maps: dict[str, list[str]] = {}

    def fit(self, corpus: pd.DataFrame) -> None:
        embeddings = np.vstack(corpus[self.emb_col].values).astype("float32")
        dim = embeddings.shape[1]
        self._global_id_map = corpus["doc_id"].tolist()

        # Global index
        self._global_index = hnswlib.Index(space="cosine", dim=dim)
        self._global_index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
        self._global_index.add_items(embeddings, list(range(len(embeddings))))
        self._global_index.set_ef(self.ef_search)

        # Per-domain indices
        for domain, group in corpus.groupby("domain"):
            if pd.isna(domain) or len(group) < 5:
                continue
            dom_embs = np.vstack(group[self.emb_col].values).astype("float32")
            dom_ids = group["doc_id"].tolist()
            idx = hnswlib.Index(space="cosine", dim=dim)
            idx.init_index(max_elements=len(dom_ids), ef_construction=200, M=16)
            idx.add_items(dom_embs, list(range(len(dom_ids))))
            idx.set_ef(self.ef_search)
            self._domain_indices[domain] = idx
            self._domain_id_maps[domain] = dom_ids

        print(f"  MultiGraph: 1 global + {len(self._domain_indices)} domain graphs")

    def _rrf_merge(self, rankings: list[tuple[list[str], float]], top_k: int) -> list[str]:
        scores: defaultdict[str, float] = defaultdict(float)
        for ranked_list, weight in rankings:
            for rank, doc_id in enumerate(ranked_list, start=1):
                scores[doc_id] += weight / (self.rrf_k + rank)
        return sorted(scores, key=scores.get, reverse=True)[:top_k]

    def retrieve(self, queries: pd.DataFrame, top_k: int = 100) -> dict[str, list[str]]:
        results = {}
        fetch_k = min(top_k * 2, len(self._global_id_map))

        for _, row in queries.iterrows():
            qid = row["doc_id"]
            q_emb = np.array(row[self.emb_col], dtype="float32").reshape(1, -1)
            q_domain = row.get("domain", None)
            rankings = []

            # Global ranking
            labels, _ = self._global_index.knn_query(q_emb, k=fetch_k)
            global_list = [self._global_id_map[i] for i in labels[0]]
            rankings.append((global_list, self.global_weight))

            # Domain ranking
            if q_domain and q_domain in self._domain_indices:
                dom_idx = self._domain_indices[q_domain]
                dom_ids = self._domain_id_maps[q_domain]
                k_dom = min(fetch_k, len(dom_ids))
                labels, _ = dom_idx.knn_query(q_emb, k=k_dom)
                dom_list = [dom_ids[i] for i in labels[0]]
                rankings.append((dom_list, self.domain_weight))

            results[qid] = self._rrf_merge(rankings, top_k)
        return results
```

---

### Approach G6: Vamana-Style Graph (RRND with α-relaxation)

Vamana uses Relaxed RND (RRND) with α>1, which retains more edges than strict RND, allowing slightly longer-range connections. Combined with a disk-resident index (DiskANN), this scales to massive corpora.

For citation retrieval at 20K, the pure in-memory Vamana graph is appropriate:

```python
# Use Microsoft's DiskANN Python bindings
# pip install diskannpy

import diskannpy
import numpy as np

# Build Vamana index (RRND with α=1.2)
builder = diskannpy.StaticMemoryIndex.build(
    data=corpus_embeddings.astype("float32"),
    distance_metric="cosine",
    index_directory="/tmp/citation_vamana",
    complexity=64,       # L parameter: beam width during construction
    graph_degree=32,     # R parameter: max out-degree
    alpha=1.2,           # RRND relaxation: α > 1 retains more long-range edges
    num_threads=8,
)

# Query
index = diskannpy.StaticMemoryIndex(
    index_directory="/tmp/citation_vamana",
    num_threads=4,
    initial_search_complexity=100,  # ef_search equivalent
)
labels, distances = index.search(query_embedding, k_neighbors=100, complexity=100)
```

**Research value**: Compare Vamana (RRND, α=1.2) vs NSG (RND, strict) vs HNSW (RRND within layer) to quantify the effect of α on citation retrieval recall. From the PDF (page 82), RRND prunes <1% of edges while RND prunes 5-20% — is the extra recall from RRND worth the memory overhead for 20K papers?

---

## The Novel Research Contribution: Citation-Graph Augmented Proximity Graph

This is the key *new approach* that merges graph-based vector search with the unique structure of the citation retrieval task.

**Observation**: The corpus has two types of relationships:
1. **Semantic proximity**: papers close in embedding space (captured by HNSW/NSG)
2. **Citation proximity**: papers that cite each other (structural graph from qrels/metadata)

**Novel Idea**: Augment the proximity graph with citation edges during construction:

```
Algorithm CitationAugmentedHNSW:

Phase 1 — Build standard HNSW on embeddings

Phase 2 — Citation edge injection:
  For each (query_paper, cited_paper) pair in qrels_train:
    IF cited_paper NOT already in HNSW neighborhood of query_paper:
      Add cited_paper as an additional neighbor with edge weight w_cite
      Apply RND pruning: only add if it passes the RNG condition
        i.e., dist(query, cited) < dist(n, cited) for all existing neighbors n

Phase 3 — Query:
  Standard HNSW beam search, but edges have:
    weight = α * semantic_sim + (1-α) * citation_edge_indicator

Intuition: Papers that are BOTH semantically similar AND cited together
are strongly connected → beam search finds them faster.
```

**Why this is novel**: No existing graph-based ANNS work injects domain-specific structural edges (citations) into the proximity graph. This creates a **heterogeneous proximity graph** that is tailored for citation retrieval.

**Expected behavior**: For queries where the cited papers are semantically distant (cross-domain citations), pure semantic HNSW fails. Citation-augmented HNSW retains those cross-domain edges → higher Recall@100.

---

## Dataset Hardness Analysis for This Challenge

From the PDF (pages 42-43), two metrics characterize dataset hardness for ANNS:

**LID (Local Intrinsic Dimensionality)**: Lower = easier. Scientific paper embeddings (all-MiniLM-L6-v2, 384-dim) projected to 384 dimensions but intrinsic dimensionality is ~50-100 → moderate difficulty.

**LRC (Local Relative Contrast)**: Higher = easier. Citations are sparse → many relevant papers may be far in embedding space → low LRC → hard dataset.

**Implication from PDF (page 87)**: Divide-and-Conquer methods (ELPIS, SPTAG) outperform HNSW on hard datasets (low LID, low LRC). Citation retrieval is structurally hard → ELPIS-style domain-partitioned graph (Approach G3) should outperform flat HNSW (Approach G1).

```python
def compute_lid(embeddings, k=10):
    """Estimate Local Intrinsic Dimensionality of the embedding space."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k+1, metric="cosine").fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    distances = distances[:, 1:]  # exclude self
    # MLE estimator for LID
    r_k = distances[:, -1]  # distance to k-th neighbor
    lid = -1.0 / np.mean(np.log(distances / r_k[:, None] + 1e-9), axis=1)
    return float(np.mean(lid))

def compute_lrc(embeddings, k=10):
    """Estimate Local Relative Contrast."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k+1, metric="cosine").fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    r_1 = distances[:, 1]   # nearest neighbor distance
    r_k = distances[:, -1]  # k-th neighbor distance
    lrc = r_k / (r_1 + 1e-9)
    return float(np.mean(lrc))
```

Run this analysis at the start of benchmarking. If LID < 30 and LRC < 3 → use ELPIS/SPTAG-style DC method. If LID > 50 and LRC > 5 → HNSW/NSG is sufficient.

---

## Benchmark Registry: Adding to run_benchmark.py

```python
# In scripts/run_benchmark.py — build_registry() function

from src.retrievers.hnsw_dense import HNSWDenseRetriever
from src.retrievers.filtered_hnsw import DomainFilteredHNSW
from src.retrievers.domain_graph import DomainPartitionedGraphRetriever
from src.retrievers.multigraph_rrf import MultiGraphRRF
from src.retrievers.nndescent_expand import NNDescentGraphExpander

def build_registry(corpus, queries):
    return {
        # --- Existing baselines ---
        "tfidf":          TFIDFRetriever(),
        "bm25":           BM25Retriever(),
        "dense_minilm":   DenseRetriever(emb_col="embedding"),
        "hybrid_bm25":    HybridRetriever(BM25Retriever(), DenseRetriever()),
        "rerank_dense":   CrossEncoderReranker(DenseRetriever()),

        # --- G1: HNSW drop-in (M sweep) ---
        "hnsw_M8":        HNSWDenseRetriever(M=8,  ef_construction=200, ef_search=100),
        "hnsw_M16":       HNSWDenseRetriever(M=16, ef_construction=200, ef_search=100),
        "hnsw_M32":       HNSWDenseRetriever(M=32, ef_construction=200, ef_search=100),
        "hnsw_M64":       HNSWDenseRetriever(M=64, ef_construction=200, ef_search=100),

        # --- G1: HNSW ef_search sweep ---
        "hnsw_ef50":      HNSWDenseRetriever(M=16, ef_search=50),
        "hnsw_ef100":     HNSWDenseRetriever(M=16, ef_search=100),
        "hnsw_ef200":     HNSWDenseRetriever(M=16, ef_search=200),
        "hnsw_ef400":     HNSWDenseRetriever(M=16, ef_search=400),

        # --- G2: Domain-filtered HNSW ---
        "domain_filtered_hnsw": DomainFilteredHNSW(M=16, ef_search=100),

        # --- G3: ELPIS-style domain-partitioned graph ---
        "domain_graph":   DomainPartitionedGraphRetriever(M=16, ef_search=100),

        # --- G4: NNDescent + expansion ---
        "nndescent":      NNDescentGraphExpander(n_neighbors=30, expansion_hops=0),
        "nndescent_exp1": NNDescentGraphExpander(n_neighbors=30, expansion_hops=1),

        # --- G5: Multi-graph RRF ---
        "multigraph_rrf": MultiGraphRRF(global_weight=0.5, domain_weight=0.5),
        "multigraph_rrf_dom_heavy": MultiGraphRRF(global_weight=0.3, domain_weight=0.7),

        # --- Combined: HNSW + BM25 via RRF (Vespa-style) ---
        "hnsw_bm25_rrf":  HybridRetriever(
            BM25Retriever(), HNSWDenseRetriever(M=16, ef_search=100)
        ),
    }
```

---

## Expected Results Table (Hypothesis)

Based on the PDF's experimental findings applied to this 20K citation corpus:

| Approach | Expected NDCG@10 | Expected Recall@100 | Build Time | Notes |
|---|---|---|---|---|
| TF-IDF | ~0.15 | ~0.30 | Fast | Lexical only |
| BM25 | ~0.18 | ~0.35 | Fast | Lexical + term freq |
| Dense (brute-force) | ~0.25 | ~0.55 | Medium | Semantic |
| **HNSW M=16** | ~0.25 | ~0.55 | Fast | Same quality as brute-force |
| **HNSW M=32** | ~0.26 | ~0.56 | Medium | Slightly better coverage |
| Hybrid BM25+Dense | ~0.28 | ~0.60 | Medium | RRF fusion |
| **HNSW + BM25 RRF** | ~0.29 | ~0.62 | Medium | Graph + lexical |
| **Domain-Filtered HNSW** | ~0.27 | ~0.58 | Medium | Domain metadata exploit |
| **Domain-Partitioned (ELPIS)** | ~0.28 | ~0.60 | Fast | DC on domain structure |
| **Multi-Graph RRF** | ~0.30 | ~0.64 | Medium | Ensemble of graphs |
| CrossEncoder rerank | ~0.32 | ~0.55 | Slow | Reranking wins precision |
| **HNSW + CrossEncoder** | ~0.33 | ~0.58 | Slow | Best precision pipeline |
| **NNDescent + Expansion** | ~0.27 | ~0.62 | Medium | Graph walk expansion |

*All predictions relative to existing baselines; actual results will vary.*

---

## Research Contributions Summary (For Paper)

| # | Contribution | Novelty | From PDF Section |
|---|---|---|---|
| C1 | First application of HNSW/NSG to citation retrieval benchmark | Applied | Paradigm 2 (II) |
| C2 | Domain-partitioned proximity graph (19 domains as natural partition) | Novel application of ELPIS/DC paradigm | Paradigm 4 (DC) |
| C3 | Citation-edge-augmented proximity graph | Novel (no prior work) | — |
| C4 | RND vs RRND vs MOND ablation on scientific paper embeddings | New experimental finding | Paradigm 3 (ND) |
| C5 | LID/LRC analysis of citation retrieval embeddings → dataset hardness | Analysis | Pages 42-43 |
| C6 | Domain-filtered search (RWalks-style predicate) via domain metadata | Applied | RWalks (PACMMOD'25) |
| C7 | Multi-graph RRF ensemble (19 domain graphs + global) | Applied | Vespa/hybrid DB pattern |
| C8 | Medoid seed selection using domain centroids | Applied | Seed Selection (MD) |

---

## Installation

```bash
pip install hnswlib pynndescent nmslib diskannpy
# or for minimal setup (just HNSW):
pip install hnswlib
```

---

## References from PDF

- Malkov, Yashunin (2016/2018). HNSW. IEEE TPAMI. [hnswlib implementation]
- Azizi, Echihabi, Palpanas (2023). ELPIS: Graph-Based Similarity Search for Scalable Data Science. PVLDB 16(6): 1548-1559.
- Azizi, Echihabi, Palpanas (2025). Graph-Based Vector Search: Experimental Evaluation of SOTA. PACMMOD 3(1).
- AitAomar, Echihabi et al. (2025). RWalks: Random Walks as Attribute Diffusers for Filtered Vector Search. PACMMOD 3(3).
- Subramanya et al. (2019). DiskANN/Vamana. NeurIPS 2019.
- Patel et al. (2024). ACORN: Performant Predicate-Agnostic Search. PACMMOD.
- Dong et al. (2011). KGraph/NNDescent. WWW 2011.
