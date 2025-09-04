from typing import List

import streamlit as st
from dotenv import load_dotenv

from prompts import SYSTEM_PROMPT
from llm_clients import chat
from utils import (
    read_uploaded_files, chunk_docs, build_faiss, search,
    gather_context, coverage_score
)

# ---------- App setup ----------
load_dotenv()
st.set_page_config(page_title="Customer Support Copilot", page_icon="🤖", layout="centered")
st.title("🤖 Customer Support Copilot (RAG)")

# ---------- Session defaults ----------
if "docs" not in st.session_state:
    st.session_state.docs: List[str] = []
if "chunks" not in st.session_state:
    st.session_state.chunks: List[str] = []
if "index" not in st.session_state:
    st.session_state.index = None
if "unanswered" not in st.session_state:
    st.session_state.unanswered: List[dict] = []

# ---------- Upload & chunk ----------
files = st.file_uploader("Upload docs (.txt, .md)", accept_multiple_files=True, type=["txt", "md"])

texts = read_uploaded_files(files)
if texts:
    st.session_state.docs = texts
    st.session_state.chunks = chunk_docs(texts, chunk_size=300, overlap=100)
    st.caption(f"Loaded {len(texts)} document(s). Chunks: {len(st.session_state.chunks)}")
else:
    if not st.session_state.docs:
        st.caption("No documents loaded yet.")

# ---------- Indexing ----------
if st.session_state.chunks:
    with st.spinner("Embedding & indexing..."):
        try:
            st.session_state.index = build_faiss(st.session_state.chunks)
        except Exception as e:
            st.session_state.index = None
            st.error(f"Indexing error: {e}")

    if st.session_state.index is not None:
        st.success(f"Indexed {len(st.session_state.chunks)} chunks.")
    else:
        st.warning("No chunks to index or indexing failed.")
else:
    st.caption("Upload docs to enable indexing.")

st.divider()

# ---------- Grounded answer ----------
q = st.text_input("💬 Ask a support question (e.g., 'How do I reset my password?')")
top_k = 5

if st.button("Answer") and q.strip():
    if (st.session_state.index is None) or (not st.session_state.chunks):
        st.warning("Please upload or load docs first.")
    else:
        try:
            scores, idx = search(st.session_state.index, q.strip(), k=top_k)
            ctx = gather_context(st.session_state.chunks, idx, max_chars=3500)
            cov = coverage_score(scores)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion:\n{q.strip()}"}
            ]

            with st.spinner("Thinking..."):
                answer = chat(messages, model="gpt-5-nano")

            if answer:
                st.subheader("Answer")
                st.write(answer)
                st.caption(f"Coverage: {cov}% • Top-K: {top_k} • Model: gpt-5-nano")

                if cov < 60:
                    st.warning("Low coverage: docs may not include this answer.")
                    st.session_state.unanswered.append({"q": q.strip(), "coverage": cov})
            else:
                st.error("Received empty response from the model.")

        except Exception as e:
            st.error(f"LLM error: {e}")

st.divider()

# ---------- Low-coverage log ----------
st.subheader("Unanswered / Low-coverage questions")
if st.session_state.unanswered:
    for item in st.session_state.unanswered[-10:]:
        st.write(f"• {item['q']}  (coverage {item['coverage']}%)")
else:
    st.caption("None yet.")
