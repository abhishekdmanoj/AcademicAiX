import os
import json
import faiss
import numpy as np
import pickle
import fitz  # PyMuPDF
import re

from embeddings.model import load_embedding_model
from embeddings.embed_chunks import embed_chunks


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

REGISTRY_PATH = os.path.join(PROJECT_ROOT, "data", "registry.json")
INDEX_PATH = os.path.join(PROJECT_ROOT, "vector_store", "faiss_syllabus.index")
METADATA_PATH = os.path.join(PROJECT_ROOT, "vector_store", "metadata_syllabus.pkl")


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def clean_text(text):
    # Remove excessive whitespace but preserve structure
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def chunk_text(text, max_chars=1000):
    """
    Balanced chunking:
    - Preserve structural breaks
    - Split by heading patterns
    - Maintain size control
    - Remove tiny/noisy sections
    """

    text = clean_text(text)

    sections = re.split(
        r'\n(?=[A-Z][A-Z\s]{3,}:)|\n\d+\s+[A-Z]',
        text
    )

    chunks = []
    current_chunk = ""

    for section in sections:
        section = section.strip()

        # 🔥 Skip tiny garbage sections
        if not section or len(section) < 100:
            continue

        if len(current_chunk) + len(section) <= max_chars:
            current_chunk += "\n" + section
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = section

    if current_chunk:
        chunks.append(current_chunk.strip())

    # 🔥 Remove very small chunks
    chunks = [c for c in chunks if len(c) > 200]

    return chunks


def build_syllabus_index():
    print("🚀 Starting syllabus indexing...")

    if not os.path.exists(REGISTRY_PATH):
        print("❌ registry.json not found.")
        return

    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)

    model = load_embedding_model()

    all_embeddings = []
    metadata = []

    for entry in registry:

        if not entry.get("is_active", False):
            continue

        university = entry["university"]
        relative_path = entry["file_path"]
        file_path = os.path.join(PROJECT_ROOT, relative_path)

        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue

        print(f"📄 Processing {university}...")

        text = extract_text_from_pdf(file_path)
        chunks = chunk_text(text)

        embeddings = embed_chunks(chunks, model)

        for chunk_text_value, embedding in zip(chunks, embeddings):
            all_embeddings.append(embedding)
            metadata.append({
                "university": university,
                "unit": chunk_text_value,
                "file_path": relative_path
            })

    if not all_embeddings:
        print("❌ No embeddings generated.")
        return

    all_embeddings = np.array(all_embeddings).astype("float32")

    faiss.normalize_L2(all_embeddings)

    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(all_embeddings)

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("✅ Syllabus index built successfully!")


if __name__ == "__main__":
    build_syllabus_index()