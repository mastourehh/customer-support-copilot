# llm_clients.py
from typing import List, Dict
import os
import numpy as np
from openai import OpenAI


def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment (.env)")
    return OpenAI(api_key=api_key)


def chat(messages: List[Dict], model: str = "gpt-5-nano") -> str:
    """
    Thin wrapper for OpenAI Chat Completions.
    Returns the assistant message text (empty string on unexpected shape).
    """
    client = _get_openai_client()
    resp = client.chat.completions.create(model=model, messages=messages)
    try:
        return resp.choices[0].message.content or ""
    except Exception:
        return ""


def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """
    Batch-embed texts; returns a row-normalized float32 matrix.
    """
    if not texts:
        return np.zeros((0, 0), dtype="float32")

    client = _get_openai_client()
    resp = client.embeddings.create(model=model, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    # Normalize so inner product ≈ cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs = vecs / norms
    return vecs
