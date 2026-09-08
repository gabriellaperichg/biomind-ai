from __future__ import annotations

import importlib
import os
import threading
from typing import Sequence

MODEL_NAME = os.getenv("BIOMIND_EMBED_MODEL", "BAAI/bge-m3")
DEVICE = os.getenv("BIOMIND_EMBED_DEVICE", "cpu")
BATCH_SIZE = max(1, int(os.getenv("BIOMIND_EMBED_BATCH_SIZE", "8")))
OFFLINE = os.getenv("BIOMIND_OFFLINE", "1") == "1"

_model = None
_lock = threading.Lock()


def get_model():
    global _model
    if _model is not None:
        return _model

    with _lock:
        if _model is None:
            module = importlib.import_module("sentence_transformers")
            cls = getattr(module, "SentenceTransformer")
            _model = cls(
                MODEL_NAME,
                local_files_only=OFFLINE,
                device=DEVICE,
            )
    return _model


def embed_documents(texts: Sequence[str]) -> list[list[float]]:
    if not texts:
        return []
    model = get_model()
    encoder = getattr(model, "encode_document", model.encode)
    vectors = encoder(
        list(texts),
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_queries(texts: Sequence[str]) -> list[list[float]]:
    if not texts:
        return []
    model = get_model()
    encoder = getattr(model, "encode_query", model.encode)
    vectors = encoder(
        list(texts),
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vectors.tolist()
