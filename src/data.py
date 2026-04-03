"""Data loading utilities for the IR Challenge benchmark."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


def load_queries(path: Path = DATA_DIR / "queries.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["ta"] = df["title"].fillna("") + " " + df["abstract"].fillna("")
    return df


def load_corpus(path: Path = DATA_DIR / "corpus.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["ta"] = df["title"].fillna("") + " " + df["abstract"].fillna("")
    return df


def load_qrels(path: Path = DATA_DIR / "qrels.json") -> dict[str, list[str]]:
    with open(path) as f:
        return json.load(f)


def load_embeddings(model_name: str = "sentence-transformers_all-MiniLM-L6-v2"):
    emb_dir = DATA_DIR / "embeddings" / model_name
    query_embs = np.load(emb_dir / "query_embeddings.npy")
    corpus_embs = np.load(emb_dir / "corpus_embeddings.npy")
    with open(emb_dir / "query_ids.json") as f:
        query_ids = json.load(f)
    with open(emb_dir / "corpus_ids.json") as f:
        corpus_ids = json.load(f)
    return query_embs, corpus_embs, query_ids, corpus_ids


def get_ta(row) -> str:
    return (row.get("title", "") or "") + " " + (row.get("abstract", "") or "")


def get_chunks(row) -> list[str]:
    """Return list of text chunks from a paper's full_text using chunk_meta."""
    full_text = row.get("full_text", "") or ""
    chunk_meta = row.get("chunk_meta", []) or []
    if not chunk_meta or not full_text:
        return [get_ta(row)]
    chunks = []
    for meta in chunk_meta:
        start, end = meta.get("start", 0), meta.get("end", len(full_text))
        chunks.append(full_text[start:end])
    return chunks if chunks else [get_ta(row)]
