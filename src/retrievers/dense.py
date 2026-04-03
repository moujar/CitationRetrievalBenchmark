"""Dense retriever using sentence-transformers embeddings."""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from .base import BaseRetriever

DATA_DIR = Path(__file__).parent.parent.parent / "data"


class DenseRetriever(BaseRetriever):
    """
    Dense retrieval via cosine similarity on sentence-transformer embeddings.

    Can use:
    - Pre-computed embeddings stored in data/embeddings/<model_name>/
    - Live encoding with a SentenceTransformer model

    Parameters
    ----------
    model_name  : HuggingFace model id or local path
    use_precomputed : try to load pre-computed embeddings first
    batch_size  : encoding batch size (used only for live encoding)
    field       : text field to encode — "ta" or "full_text"
    """

    name = "Dense"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_precomputed: bool = True,
        batch_size: int = 256,
        field: str = "ta",
    ):
        self.model_name = model_name
        self.use_precomputed = use_precomputed
        self.batch_size = batch_size
        self.field = field
        self.name = f"Dense ({model_name.split('/')[-1]})"

        self._corpus_ids: list[str] = []
        self._corpus_embs: np.ndarray | None = None
        self._query_embs: np.ndarray | None = None
        self._query_ids: list[str] | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _precomputed_dir(self) -> Path:
        safe = self.model_name.replace("/", "_")
        return DATA_DIR / "embeddings" / safe

    def _load_precomputed(self):
        d = self._precomputed_dir()
        if not d.exists():
            return False
        try:
            self._corpus_embs = np.load(d / "corpus_embeddings.npy")
            self._query_embs = np.load(d / "query_embeddings.npy")
            with open(d / "corpus_ids.json") as f:
                self._corpus_ids = json.load(f)
            with open(d / "query_ids.json") as f:
                self._query_ids = json.load(f)
            return True
        except FileNotFoundError:
            return False

    def _encode(self, texts: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.model_name)
        embs = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embs.astype(np.float32)

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def fit(self, corpus: pd.DataFrame) -> None:
        if self.use_precomputed and self._load_precomputed():
            print(f"  Loaded pre-computed embeddings from {self._precomputed_dir()}")
            return

        print(f"  Encoding corpus with {self.model_name} ...")
        texts = corpus[self.field].fillna("").tolist()
        self._corpus_ids = corpus["doc_id"].tolist()
        self._corpus_embs = self._encode(texts)

    def retrieve(self, queries: pd.DataFrame, top_k: int = 100) -> dict[str, list[str]]:
        # Resolve query embeddings
        if self._query_embs is not None and self._query_ids is not None:
            # Align order to the provided queries DataFrame
            id2idx = {qid: i for i, qid in enumerate(self._query_ids)}
            query_embs = np.stack([
                self._query_embs[id2idx[qid]]
                for qid in queries["doc_id"]
                if qid in id2idx
            ])
            valid_qids = [qid for qid in queries["doc_id"] if qid in id2idx]
        else:
            print(f"  Encoding queries with {self.model_name} ...")
            texts = queries[self.field].fillna("").tolist()
            query_embs = self._encode(texts)
            valid_qids = queries["doc_id"].tolist()

        # Batch dot-product similarity (cosine on L2-normalized vectors)
        scores_matrix = query_embs @ self._corpus_embs.T  # (n_queries, n_corpus)

        results = {}
        for i, qid in enumerate(valid_qids):
            scores = scores_matrix[i]
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            ranked = [self._corpus_ids[j] for j in top_indices if self._corpus_ids[j] != qid]
            results[qid] = ranked[:top_k]

        return results
