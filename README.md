# IR Challenge Retrieval Benchmark

Benchmarking of retrieval approaches for the **scientific citation retrieval** challenge.

Given a query paper, retrieve the most relevant documents from a corpus of **20,000 scientific papers** based on citation relationships (100 queries, 19 domains, Semantic Scholar).

Hard Domain Filter + 7-Signal Ensemble: <a href="https://colab.research.google.com/drive/1fdq1odhoi3zWFtLd6yXkEI_cSZwgZo1X?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" />
  </a>
Experimente : <a href="https://github.com/moujar/CitationRetrievalBenchmark/blob/main/notebooks/experimental_research.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" />
  </a>
---

## Leaderboard

> Run `python scripts/run_benchmark.py` to reproduce.

## Summary Table

| # | Approach | Training NDCG@10  |
|---|----------|:----------------:|
| 1 | BM25 on Title+Abstract | 0.4663 |
| 2 | TF-IDF with bigrams | 0.4724 |
| 3 | MiniLM semantic search | 0.5073 |
| 4 | BM25 on body chunks | 0.5197 |
| 5 | RRF fusion (BM25 + MiniLM) | 0.5534 |
| 6 | Hard domain filter + 5 signals | 0.6398  |
| 7 | Hard domain + 6 signals | 0.6572 |
| 8 | 6 signals + CV-tuned weights | 0.6937  |
| 9 | Fine-tuned MiniLM + 7 signals | 0.7018 | 
| 10 | CE v1 reranking (rk=30) | 0.7666 |
| 11 | CE v1 + interpolation (α=0.8) | 0.7914 | 
| 12 | CE v1 + interpolation (α=0.4) | 0.8001 | 
| 13 | CE v1 + interpolation (α=0.45) | 0.8059 | 
| 14 | **CE v2 + interpolation (α=0.5)** | **0.8675** |

Fill in the table by running the benchmark and pasting your results.

---

## Project Structure

```
.
├── src/
│   ├── data.py            # Data loading utilities
│   ├── evaluate.py        # Evaluation metrics (NDCG, Recall, MRR, MAP, ...)
│   └── retrievers/
│       ├── base.py        # Abstract BaseRetriever interface
│       ├── tfidf.py       # TF-IDF sparse retrieval
│       ├── bm25.py        # BM25 sparse retrieval
│       ├── dense.py       # Dense retrieval (sentence-transformers)
│       ├── hybrid.py      # Hybrid: Reciprocal Rank Fusion (RRF)
│       └── reranker.py    # Cross-encoder two-stage reranking
├── scripts/
│   ├── run_benchmark.py   # Main benchmark runner + leaderboard printer
│   └── embed.py           # Pre-compute embeddings for any model
├── data/                  # (not committed) corpus, queries, qrels, embeddings
├── results/               # Saved ranked outputs (JSON)
└── notebooks/
    └── challenge.ipynb    # Exploratory analysis + baseline notebooks
```

---

## Setup

```bash
pip install -r requirements.txt
```

Data files (`corpus.parquet`, `queries.parquet`, `qrels.json`) and pre-computed embeddings are not included in this repo. Place them under `data/` following the structure above.

---

## Usage

### Run all retrievers

```bash
python scripts/run_benchmark.py
```

### Run specific retrievers

```bash
python scripts/run_benchmark.py --retrievers tfidf bm25 dense_minilm
```

### Save results + per-domain breakdown

```bash
python scripts/run_benchmark.py --save --domains
```

### Available retriever keys

| Key | Description |
|---|---|
| `tfidf` | TF-IDF on title + abstract |
| `bm25` | BM25 on title + abstract |
| `dense_minilm` | Dense: `all-MiniLM-L6-v2` (pre-computed) |
| `dense_bge` | Dense: `BAAI/bge-small-en-v1.5` |
| `hybrid_bm25` | BM25 + Dense MiniLM via RRF |
| `hybrid_tfidf` | TF-IDF + Dense MiniLM via RRF |
| `rerank_bm25` | BM25 → CrossEncoder reranking |
| `rerank_dense` | Dense MiniLM → CrossEncoder reranking |

### Pre-compute embeddings for a new model

```bash
python scripts/embed.py --model BAAI/bge-small-en-v1.5
```

---

## Adding a New Retriever

1. Create `src/retrievers/my_retriever.py` inheriting `BaseRetriever`:

```python
from .base import BaseRetriever

class MyRetriever(BaseRetriever):
    name = "My Retriever"

    def fit(self, corpus):
        ...  # index the corpus

    def retrieve(self, queries, top_k=100):
        ...  # return {query_id: [ranked doc_ids]}
        return results
```

2. Register it in `scripts/run_benchmark.py` inside `build_registry()`:

```python
"my_retriever": MyRetriever(),
```

3. Run:

```bash
python scripts/run_benchmark.py --retrievers my_retriever
```

---

## Evaluation Metrics

All metrics are computed over 100 public queries with citation-based ground truth.

| Metric | Description |
|---|---|
| NDCG@10 / @100 | Normalized Discounted Cumulative Gain |
| Recall@10 / @100 | Fraction of gold docs retrieved in top-k |
| Precision@10 | Precision at rank 10 |
| MRR@10 | Mean Reciprocal Rank |
| MAP | Mean Average Precision |

---

## Contributing

Pull requests are welcome! To contribute a new retriever:
- Follow the `BaseRetriever` interface
- Add it to the registry in `run_benchmark.py`
- Fill in the leaderboard table with your results
