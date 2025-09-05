from typing import List, Tuple
import faiss

from llm_clients import embed_texts


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 100) -> List[str]:
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
    all_chunks: List[str] = []
    for doc in texts:
        parts = chunk_text(doc, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(parts)
    return all_chunks


def read_uploaded_files(files) -> List[str]:
    texts: List[str] = []
    for f in (files or []):
        raw = f.read()
        txt = raw.decode("utf-8", errors="ignore")
        if txt.strip():
            texts.append(txt)
    return texts


def build_faiss(chunks: List[str]):
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
    if index is None or not query.strip():
        return [], []

    qv = embed_texts([query])
    if qv.size == 0:
        return [], []

    D, I = index.search(qv.astype("float32"), k)
    return D[0].tolist(), I[0].tolist()


def gather_context(chunks: List[str], idx: List[int], max_chars: int = 3500) -> str:
    pieces = [chunks[i] for i in idx if 0 <= i < len(chunks)]
    ctx = "\n\n---\n\n".join(pieces)
    return ctx[:max_chars]


def coverage_score(scores: List[float]) -> float:
    if not scores:
        return 0.0
    top = max(scores)
    pct = max(0.0, min(1.0, (top + 1) / 2)) * 100
    return round(pct, 1)
