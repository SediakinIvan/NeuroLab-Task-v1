from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import EmbeddingsConfig
from src.io_utils import write_json


@dataclass(frozen=True)
class EmbeddingResult:
    embeddings_path: Path
    metadata_path: Path


def _encode_batches(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    normalize_embeddings: bool,
    show_progress_bar: bool,
    checkpoint_every: int,
    checkpoint_prefix: Path,
) -> np.ndarray:
    all_batches: list[np.ndarray] = []
    iterator = range(0, len(texts), batch_size)
    if show_progress_bar:
        iterator = tqdm(iterator, desc="Embedding text batches")

    for idx, start in enumerate(iterator, start=1):
        end = min(start + batch_size, len(texts))
        chunk = texts[start:end]
        emb = model.encode(
            chunk,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        all_batches.append(emb)

        if checkpoint_every > 0 and idx % checkpoint_every == 0:
            partial = np.vstack(all_batches)
            np.save(checkpoint_prefix.with_name(f"{checkpoint_prefix.name}_partial_{idx}.npy"), partial)

    if not all_batches:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(all_batches)


def build_embeddings(
    text_df: pd.DataFrame,
    text_column: str,
    config: EmbeddingsConfig,
    models_dir: Path,
    reports_dir: Path,
) -> EmbeddingResult:
    if text_column not in text_df.columns:
        raise ValueError(f"Embedding column '{text_column}' not found in text dataset.")

    texts = text_df[text_column].fillna("").astype(str).tolist()
    model = SentenceTransformer(config.model_name)

    checkpoint_prefix = models_dir / "text_embeddings_checkpoint"
    embeddings = _encode_batches(
        model=model,
        texts=texts,
        batch_size=config.batch_size,
        normalize_embeddings=config.normalize_embeddings,
        show_progress_bar=config.show_progress_bar,
        checkpoint_every=config.checkpoint_every,
        checkpoint_prefix=checkpoint_prefix,
    )

    embeddings_path = models_dir / "text_embeddings.npy"
    np.save(embeddings_path, embeddings)

    metadata_payload: dict[str, Any] = {
        "model_name": config.model_name,
        "rows_embedded": int(embeddings.shape[0]),
        "embedding_dimension": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.shape[0] > 0 else 0,
        "batch_size": config.batch_size,
        "normalize_embeddings": config.normalize_embeddings,
        "checkpoint_every": config.checkpoint_every,
        "source_text_column": text_column,
    }
    metadata_path = reports_dir / "embedding_meta.json"
    write_json(metadata_path, metadata_payload)

    return EmbeddingResult(embeddings_path=embeddings_path, metadata_path=metadata_path)
