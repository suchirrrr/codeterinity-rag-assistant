"""
codeterinity.py

Terminal chat app for CodeTerinity.

Features:
- Connects to your local LM Studio server.
- Reads context from ChromaDB collection built by build_memory.py.
- Uses PDF memory when it is relevant.
- Falls back to general knowledge when PDFs do not help.
"""

from typing import List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ---------- CONFIG ----------
DB_DIR = r"C:\AI\assistant\db"
COLLECTION_NAME = "codeterinity_memory"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
DISTANCE_THRESHOLD = 1.2  # higher = easier to use PDF context (0.8–1.5 typical)

LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
LMSTUDIO_API_KEY = "lm-studio"
LMSTUDIO_MODEL = "phi-3.1-mini-4k-instruct"  # must match LM Studio "API identifier"
# ----------------------------

_embedder = None
_collection = None
_client = None


def get_embedder() -> SentenceTransformer:
    """Lazy-load the sentence-transformer model (fast after first load)."""
    global _embedder
    if _embedder is None:
        print(f"🧠 Loading embedding model: {EMBED_MODEL_NAME} ...")
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def get_collection():
    """Lazy-connect to ChromaDB."""
    global _collection
    if _collection is None:
        print(f"🗂️  Connecting to Chroma at: {DB_DIR}")
        client = chromadb.PersistentClient(path=DB_DIR)
        _collection = client.get_collection(name=COLLECTION_NAME)
    return _collection


def get_lm_client() -> OpenAI:
    """Lazy-connect to LM Studio."""
    global _client
    if _client is None:
        print(f"🤖 Connecting to LM Studio at {LMSTUDIO_BASE_URL} ...")
        _client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)
    return _client


def retrieve_relevant_chunks(question: str, top_k: int = TOP_K) -> Tuple[str, float]:
    """
    Retrieve text chunks relevant to the question.

    Returns:
        context_text: formatted string with top chunks (possibly empty).
        best_distance: smallest distance among retrieved chunks.
    """
    embedder = get_embedder()
    collection = get_collection()

    q_emb = embedder.encode([question]).tolist()

    results = collection.query(
        query_embeddings=q_emb,
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )

    docs: List[str] = results.get("documents", [[]])[0]
    distances: List[float] = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not docs:
        return "", 999.0

    best_distance = min(distances) if distances else 999.0

    # If everything is far away → treat as "no useful PDF context"
    if best_distance > DISTANCE_THRESHOLD:
        return "", best_distance

    # Otherwise, build a context string from the top 2 chunks
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
    """Generate an answer using LM Studio, with PDF memory if relevant."""
    context_text, best_distance = retrieve_relevant_chunks(question)
    client = get_lm_client()

    if context_text:
        # Document-aware mode
        system_prompt = (
            "You are CodeTerinity, a helpful AI assistant.\n"
            "You sometimes receive extra CONTEXT from the user's PDF documents.\n"
            "When context is provided, USE IT as your main source of truth.\n"
            "You can summarise, explain, and rephrase the document.\n"
            "If something is NOT in the context, you may still answer using your\n"
            "general knowledge, but never contradict the document.\n"
        )

        user_prompt = (
            "CONTEXT FROM PDF DOCUMENTS:\n"
            f"{context_text}\n\n"
            "USER QUESTION:\n"
            f"{question}\n\n"
            "If the question is clearly about this document (for example MindEngine "
            "Expo 2025), explain it in simple normal words using the context above.\n"
        )
    else:
        # No good context → normal local assistant
        system_prompt = (
            "You are CodeTerinity, a friendly local AI assistant running on the user's PC.\n"
            "You do NOT have any useful document context for this question right now.\n"
            "Answer based on your own knowledge. If you truly don't know, say so honestly.\n"
            "You do not have live internet access, so don't pretend to 'search Google'.\n"
        )
        user_prompt = question

    response = client.chat.completions.create(
        model=LMSTUDIO_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.35,
        max_tokens=800,
    )

    return response.choices[0].message.content.strip()


def main():
    print("=" * 60)
    print("📘 CodeTerinity — Terminal Chat")
    print("Type your question and press Enter.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    while True:
        try:
            user_q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye from CodeTerinity 👋")
            break

        if user_q.lower() in {"exit", "quit", "q"}:
            print("Bye from CodeTerinity 👋")
            break

        if not user_q:
            continue

        answer = answer_with_memory(user_q)
        print("\nAI:", answer)
        print("\n" + "-" * 60)


if __name__ == "__main__":
    main()
