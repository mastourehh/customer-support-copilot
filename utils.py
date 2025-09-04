from typing import List, Tuple
import faiss

from llm_clients import embed_texts


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 100) -> List[str]:
    """
    Sliding window over words with overlap.
    chunk_size/overlap are in *words* (not characters).
    """
    words = text.split()
    chunks: List[str] = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += step
    return chunks


def chunk_docs(texts: List[str], chunk_size: int = 300, overlap: int = 100) -> List[str]:
    """Chunk multiple documents and return a single list of chunks."""
    all_chunks: List[str] = []
    for doc in texts:
        parts = chunk_text(doc, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(parts)
    return all_chunks


def read_uploaded_files(files) -> List[str]:
    """
    Streamlit uploader 'files' -> list of decoded text strings.
    Keeps logic here to keep app.py tidy (but avoids importing streamlit).
    """
    texts: List[str] = []
    for f in (files or []):
        raw = f.read()
        txt = raw.decode("utf-8", errors="ignore")
        if txt.strip():
            texts.append(txt)
    return texts


def build_faiss(chunks: List[str]):
    """
    Embed chunks and return a FAISS IndexFlatIP with those embeddings added.
    Returns None if no chunks.
    """
    if not chunks:
        return None

    vecs = embed_texts(chunks)
    if vecs.size == 0:
        return None

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs.astype("float32"))
    return index


def search(index, query: str, k: int = 5) -> Tuple[List[float], List[int]]:
    """
    Embed a query and retrieve top-k results from FAISS index.
    Returns (scores, indices).
    """
    if index is None or not query.strip():
        return [], []

    qv = embed_texts([query])
    if qv.size == 0:
        return [], []

    D, I = index.search(qv.astype("float32"), k)
    return D[0].tolist(), I[0].tolist()


def gather_context(chunks: List[str], idx: List[int], max_chars: int = 3500) -> str:
    """Join selected chunks with separators; clamp length."""
    pieces = [chunks[i] for i in idx if 0 <= i < len(chunks)]
    ctx = "\n\n---\n\n".join(pieces)
    return ctx[:max_chars]


def coverage_score(scores: List[float]) -> float:
    """
    Heuristic: map top inner product ([-1, 1]) to 0..100%.
    """
    if not scores:
        return 0.0
    top = max(scores)
    pct = max(0.0, min(1.0, (top + 1) / 2)) * 100
    return round(pct, 1)
