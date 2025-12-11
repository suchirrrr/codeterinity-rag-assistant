"""
ui_codeterinity.py

Streamlit web UI for CodeTerinity.
- Chat interface.
- Optional PDF upload to extend memory.
- Uses the same ChromaDB + LM Studio pipeline as codeterinity.py.
"""

from pathlib import Path
from typing import List

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from openai import OpenAI

# ------------ CONFIG ------------
ROOT_DIR = Path(r"C:\AI\assistant")
DATA_DIR = ROOT_DIR / "data"
DB_DIR = ROOT_DIR / "db"
COLLECTION_NAME = "codeterinity_memory"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
DISTANCE_THRESHOLD = 1.2

LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
LMSTUDIO_API_KEY = "lm-studio"
LMSTUDIO_MODEL = "phi-3.1-mini-4k-instruct"

LOGO_PATH = ROOT_DIR / "CodeTerinity.png"
# -------------------------------


# ---------- Helper singletons ----------

def get_embedder():
    if "embedder" not in st.session_state:
        st.session_state.embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return st.session_state.embedder


def get_collection():
    if "collection" not in st.session_state:
        client = chromadb.PersistentClient(path=str(DB_DIR))
        st.session_state.collection = client.get_or_create_collection(
            name=COLLECTION_NAME
        )
    return st.session_state.collection


def get_lm_client():
    if "lm_client" not in st.session_state:
        st.session_state.lm_client = OpenAI(
            base_url=LMSTUDIO_BASE_URL,
            api_key=LMSTUDIO_API_KEY,
        )
    return st.session_state.lm_client


# ---------- Chunking & indexing ----------

def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= length:
            break

        start = max(end - overlap, end - chunk_size // 2)

    return chunks


def index_uploaded_pdfs(files, clear_existing: bool = True):
    """
    Index uploaded PDFs directly into Chroma.

    files: list of UploadedFile from Streamlit.
    """
    collection = get_collection()
    embedder = get_embedder()

    if clear_existing:
        existing = collection.get()
        ids = existing.get("ids", [])
        if ids:
            collection.delete(ids=ids)

    total_chunks = 0

    for uploaded in files:
        filename = getattr(uploaded, "name", "uploaded.pdf")
        uploaded.seek(0)
        reader = PdfReader(uploaded)

        all_chunks: List[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            all_chunks.extend(chunk_text(text))

        if not all_chunks:
            continue

        embeddings = embedder.encode(all_chunks, show_progress_bar=False).tolist()
        ids = [f"{filename}-{i}" for i in range(len(all_chunks))]
        metadatas = [
            {"source": filename, "chunk_index": i}
            for i in range(len(all_chunks))
        ]

        collection.add(
            ids=ids,
            documents=all_chunks,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        total_chunks += len(all_chunks)

    return total_chunks


# ---------- Retrieval & answer ----------

def retrieve_relevant_chunks(question: str):
    embedder = get_embedder()
    collection = get_collection()

    q_emb = embedder.encode([question]).tolist()

    results = collection.query(
        query_embeddings=q_emb,
        n_results=TOP_K,
        include=["documents", "distances", "metadatas"],
    )

    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not docs:
        return "", 999.0

    best_distance = min(distances) if distances else 999.0
    if best_distance > DISTANCE_THRESHOLD:
        return "", best_distance

    context_parts = []
    for doc, dist, meta in list(zip(docs, distances, metadatas))[:2]:
        src = meta.get("source", "unknown.pdf")
        idx = meta.get("chunk_index", "?")
        context_parts.append(
            f"[Source: {src}, chunk {idx}, distance={dist:.3f}]\n{doc}"
        )

    context_text = "\n\n---\n\n".join(context_parts)
    return context_text, best_distance


def answer_with_memory(question: str) -> str:
    context_text, best_distance = retrieve_relevant_chunks(question)
    client = get_lm_client()

    if context_text:
        system_prompt = (
            "You are CodeTerinity, a helpful AI tutor.\n"
            "You sometimes receive extra CONTEXT from the user's PDFs.\n"
            "Use the context as your primary reference when it is provided.\n"
            "Explain concepts in simple, clear language.\n"
        )
        prompt_content = (
            "CONTEXT FROM DOCUMENTS:\n"
            f"{context_text}\n\n"
            "USER QUESTION:\n"
            f"{question}\n"
        )
    else:
        system_prompt = (
            "You are CodeTerinity, a helpful personal AI assistant.\n"
            "You have no useful document context for this question right now.\n"
            "Answer using your own knowledge. You do not have live internet access.\n"
        )
        prompt_content = question

    response = client.chat.completions.create(
        model=LMSTUDIO_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_content},
        ],
        temperature=0.35,
        max_tokens=800,
    )

    return response.choices[0].message.content.strip()


# -------------------- UI --------------------

st.set_page_config(page_title="CodeTerinity", page_icon="📘")

# Top bar with logo + title
cols = st.columns([1, 4])
with cols[0]:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_column_width=True)
with cols[1]:
    st.title("CodeTerinity")
    st.caption("NEW BEGINNINGS · Your local lecture-aware AI")

st.sidebar.header("Status")
st.sidebar.write(f"📂 Data folder: `{DATA_DIR}`")
st.sidebar.write(f"🧠 DB folder: `{DB_DIR}`")
st.sidebar.write(f"📚 Collection: `{COLLECTION_NAME}`")

# File upload section
st.sidebar.markdown("### 📄 Upload PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDFs to index",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files and st.sidebar.button("Index uploaded PDFs (replace existing)"):
    with st.spinner("Indexing PDFs into memory..."):
        total = index_uploaded_pdfs(uploaded_files, clear_existing=True)
    st.sidebar.success(f"Indexed {len(uploaded_files)} file(s), {total} chunks total.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I'm **CodeTerinity**.\n\n"
                "- If you upload PDFs in the sidebar, I can use them to answer questions.\n"
                "- If not, I’ll still try to help using my own knowledge.\n\n"
                "What would you like to work on today?"
            ),
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me about your lecture or anything else..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = answer_with_memory(prompt)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
