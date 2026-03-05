import os
import json
import faiss
import numpy as np
import pickle
import fitz
import re
import hashlib

from embeddings.model import load_embedding_model
from embeddings.embed_chunks import embed_chunks

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

REGISTRY_PATH = os.path.join(PROJECT_ROOT, "data", "registry.json")
CHAT_INDEX_PATH = os.path.join(PROJECT_ROOT, "vector_store", "faiss_chat.index")
CHAT_METADATA_PATH = os.path.join(PROJECT_ROOT, "vector_store", "metadata_chat.pkl")


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def clean_text(text):
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def is_generic_chunk(text):
    generic_keywords = [
        "vision", "mission", "articulation matrix",
        "list of electives", "total credits",
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in generic_keywords)


def chunk_text(text, max_chars=800):
    text = clean_text(text)
    sections = re.split(
        r"\n(?=[A-Z][A-Z\s]{3,}:)|\n\d+\s+[A-Z]",
        text
    )

    chunks = []
    current_chunk = ""

    for section in sections:
        section = section.strip()
        if not section or len(section) < 80:
            continue
        if len(current_chunk) + len(section) <= max_chars:
            current_chunk += "\n" + section
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = section

    if current_chunk:
        chunks.append(current_chunk.strip())

    return [c for c in chunks if len(c) > 80 and not is_generic_chunk(c)]


def build_chat_index():
    print("=" * 60)
    print("Building CHUNK-LEVEL chat index...")
    print("=" * 60)

    if not os.path.exists(REGISTRY_PATH):
        print("ERROR: registry.json not found.")
        return

    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)

    model = load_embedding_model()

    all_vectors = []
    all_metadata = []

    for entry in registry:
        if not entry.get("is_active", False):
            continue

        college = entry["college"]
        program = entry["program"]
        relative_path = entry["file_path"]
        file_path = os.path.join(PROJECT_ROOT, relative_path)

        if not os.path.exists(file_path):
            print(f"WARNING: File not found -> {file_path}")
            continue

        print(f"\nProcessing: {college} - {program}")

        text = extract_text_from_pdf(file_path)
        chunks = chunk_text(text)

        if not chunks:
            print(f"  WARNING: No chunks found, skipping.")
            continue

        print(f"  {len(chunks)} chunks found")

        embeddings = embed_chunks(chunks, model)
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)

        for i, (vec, chunk_text_val) in enumerate(zip(embeddings, chunks)):
            all_vectors.append(vec)
            all_metadata.append({
                "college": college,
                "program": program,
                "file_path": relative_path,
                "text": chunk_text_val,
                "chunk_id": i
            })

    if not all_vectors:
        print("ERROR: No vectors generated.")
        return

    all_vectors = np.array(all_vectors).astype("float32")
    dimension = all_vectors.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(all_vectors)

    os.makedirs(os.path.dirname(CHAT_INDEX_PATH), exist_ok=True)

    faiss.write_index(index, CHAT_INDEX_PATH)
    with open(CHAT_METADATA_PATH, "wb") as f:
        pickle.dump(all_metadata, f)

    print(f"\n✅ Chat index built: {len(all_vectors)} chunks across {len(registry)} programs")
    print("=" * 60)


if __name__ == "__main__":
    build_chat_index()
