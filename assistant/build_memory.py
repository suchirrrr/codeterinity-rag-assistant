"""
build_memory.py

Builds a ChromaDB collection from all PDFs in C:\\AI\\assistant\\data.
- Chunks each PDF.
- Encodes chunks with SentenceTransformer.
- Stores in a persistent Chroma DB at C:\\AI\\assistant\\db.
"""

from pathlib import Path
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# -------- CONFIG --------
DATA_DIR = Path(r"C:\AI\assistant\data")
DB_DIR = Path(r"C:\AI\assistant\db")
COLLECTION_NAME = "codeterinity_memory"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
# ------------------------


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping chunks safely (no infinite loops).

    Example: text of length 5000, chunk_size 700, overlap 100
    → 700 chars, then jump forward (600), etc.
    """
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
            break  # reached the end safely

        # Move forward with some overlap, but NEVER backwards
        # (protects you from infinite loops + crazy overlap)
        start = max(end - overlap, end - chunk_size // 2)

    return chunks


def get_or_create_collection():
    """Return the Chroma collection (we embed manually, so no embedding_function)."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    print(f"🧠 Using ChromaDB at: {DB_DIR}")

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def clear_collection(collection) -> None:
    """Delete all existing chunks in the collection."""
    existing = collection.get()
    ids = existing.get("ids", [])

    if ids:
        print(f"🧹 Clearing {len(ids)} existing chunks from collection...")
        collection.delete(ids=ids)
    else:
        print("🧹 Collection is already empty.")


def build_memory(clear_old: bool = True) -> None:
    """Build / rebuild the memory from all PDFs in DATA_DIR."""
    print("📚 Building memory for CodeTerinity...")
    print(f"📂 Scanning folder: {DATA_DIR}")

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")

    collection = get_or_create_collection()

    if clear_old:
        clear_collection(collection)

    print(f"\n🧠 Loading embedding model: {EMBED_MODEL_NAME} ...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print("⚠️ No PDFs found in the data folder.")
        return

    total_chunks = 0

    for pdf_path in pdf_files:
        print(f"\n📄 Processing: {pdf_path.name}")
        reader = PdfReader(str(pdf_path))

        all_chunks: List[str] = []
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            page_chunks = chunk_text(page_text)
            all_chunks.extend(page_chunks)

        if not all_chunks:
            print("   ⚠️ No extractable text in this PDF, skipping.")
            continue

        print(f"  ✂️ Created {len(all_chunks)} text chunks.")

        ids = [f"{pdf_path.stem}-{i}" for i in range(len(all_chunks))]
        metadatas = [
            {"source": pdf_path.name, "chunk_index": i}
            for i in range(len(all_chunks))
        ]

        print("  🔍 Computing embeddings for chunks...")
        embeddings = embedder.encode(all_chunks, show_progress_bar=False).tolist()

        print("  💾 Saving chunks to ChromaDB...")
        collection.add(
            ids=ids,
            documents=all_chunks,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        total_chunks += len(all_chunks)

    print("\n✅ Memory built successfully for all PDFs in the data folder!")
    print(f"   Total chunks stored: {total_chunks}")
    print(f"   DB path: {DB_DIR}")
    print(f"   Collection: {COLLECTION_NAME}")


def main():
    build_memory(clear_old=True)


if __name__ == "__main__":
    main()
