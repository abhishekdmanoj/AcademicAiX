import os
import json
import faiss
import numpy as np
import pickle
import fitz  # PyMuPDF
import re
import hashlib
from datetime import datetime

from embeddings.model import load_embedding_model
from embeddings.embed_chunks import embed_chunks


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

REGISTRY_PATH = os.path.join(PROJECT_ROOT, "data", "registry.json")
INDEX_PATH = os.path.join(PROJECT_ROOT, "vector_store", "faiss_syllabus.index")
METADATA_PATH = os.path.join(PROJECT_ROOT, "vector_store", "metadata_syllabus.pkl")


def compute_sha256(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def clean_text(text):
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def is_generic_chunk(text):
    generic_keywords = [
        "vision", "mission", "program outcomes", "peo", "po-",
        "scheme of", "curriculum", "credits", "evaluation",
        "project phase", "articulation matrix",
        "programme outcomes", "program educational objectives",
        "list of electives", "total credits"
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in generic_keywords)


def chunk_text(text, max_chars=1000):
    text = clean_text(text)

    sections = re.split(
        r'\n(?=[A-Z][A-Z\s]{3,}:)|\n\d+\s+[A-Z]',
        text
    )

    chunks = []
    current_chunk = ""

    for section in sections:
        section = section.strip()
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

    chunks = [
        c for c in chunks
        if len(c) > 200 and not is_generic_chunk(c)
    ]

    return chunks


def build_syllabus_index():
    print("🚀 Building PROGRAM-LEVEL syllabus index...")

    if not os.path.exists(REGISTRY_PATH):
        print("❌ registry.json not found.")
        return

    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)

    model = load_embedding_model()

    program_vectors = []
    metadata = []

    for entry in registry:

        if not entry.get("is_active", False):
            continue

        college = entry["college"]
        program = entry["program"]

        relative_path = entry["file_path"]
        file_path = os.path.join(PROJECT_ROOT, relative_path)

        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue

        print(f"📄 Processing {college} - {program}")

        new_hash = compute_sha256(file_path)
        entry["hash"] = new_hash
        entry["last_checked"] = str(datetime.now().date())

        text = extract_text_from_pdf(file_path)
        chunks = chunk_text(text)

        if not chunks:
            continue

        embeddings = embed_chunks(chunks, model)
        embeddings = np.array(embeddings).astype("float32")

        # Normalize unit embeddings
        faiss.normalize_L2(embeddings)

        # Compute centroid vector
        centroid = np.mean(embeddings, axis=0).astype("float32")
        centroid = centroid.reshape(1, -1)

        # Normalize centroid
        faiss.normalize_L2(centroid)

        program_vectors.append(centroid[0])

        metadata.append({
            "college": college,
            "program": program,
            "file_path": relative_path
        })

    if not program_vectors:
        print("❌ No program vectors generated.")
        return

    program_vectors = np.array(program_vectors).astype("float32")

    dimension = program_vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(program_vectors)

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    print("✅ PROGRAM-LEVEL syllabus index built successfully!")


if __name__ == "__main__":
    build_syllabus_index()