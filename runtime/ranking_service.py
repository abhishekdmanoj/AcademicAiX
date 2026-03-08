import faiss
import json
import os
import pickle
import numpy as np
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UNIVERSITY_METADATA_PATH = os.path.join(PROJECT_ROOT, "data", "university_metadata.json")
REGISTRY_PATH = os.path.join(PROJECT_ROOT, "data", "registry.json")
CHAT_INDEX_PATH = os.path.join(PROJECT_ROOT, "vector_store", "faiss_chat.index")
CHAT_METADATA_PATH = os.path.join(PROJECT_ROOT, "vector_store", "metadata_chat.pkl")

# Load chunk index once at module level
_chunk_index = None
_chunk_meta = None

def load_chunk_index():
    global _chunk_index, _chunk_meta
    if _chunk_index is not None:
        return
    if not os.path.exists(CHAT_INDEX_PATH) or not os.path.exists(CHAT_METADATA_PATH):
        print("Warning: chunk index not found, top_units will be empty")
        return
    _chunk_index = faiss.read_index(CHAT_INDEX_PATH)
    with open(CHAT_METADATA_PATH, "rb") as f:
        _chunk_meta = pickle.load(f)
    print(f"Chunk index loaded for ranking: {len(_chunk_meta)} chunks")

load_chunk_index()


def load_university_metadata():
    if not os.path.exists(UNIVERSITY_METADATA_PATH):
        return {}
    with open(UNIVERSITY_METADATA_PATH, "r") as f:
        return json.load(f)


def load_registry():
    if not os.path.exists(REGISTRY_PATH):
        return []
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def classify_alignment(score):
    if score >= 0.30:
        return "Strong"
    elif score >= 0.20:
        return "Moderate"
    else:
        return "Weak"


def normalize(text):
    if not text:
        return ""
    return text.strip().lower()




def clean_chunk(text):
    text = text.replace("\n", " ")
    # Remove lines that are mostly numbers/symbols (table noise)
    text = re.sub(r'\b[\d\s\-]{10,}\b', ' ', text)
    # Remove isolated single characters at start
    text = re.sub(r'^[A-Za-z0-9]\s*[-–]\s*', '', text.strip())
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()



def get_top_units(query_vector, college, program, top_k=3):
    """
    Query chunk-level index filtered to a specific college+program.
    Returns top_k chunks with their text and similarity score.
    """
    if _chunk_index is None or _chunk_meta is None:
        return []

    try:
        # Search broadly then filter
        search_k = min(300, len(_chunk_meta))
        distances, indices = _chunk_index.search(
            query_vector.astype("float32"), search_k
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(_chunk_meta):
                continue
            meta = _chunk_meta[idx]
            if (normalize(meta.get("college", "")) == normalize(college) and
                    normalize(meta.get("program", "")) == normalize(program)):
                results.append({
                    "unit": clean_chunk(meta.get("text", ""))[:300],
                    "similarity": round(float(dist), 4)
                })
            if len(results) >= top_k:
                break

        return results

    except Exception as e:
        print(f"top_units error: {e}")
        return []


def rank_universities(
    interest,
    model,
    index,
    metadata,
    country=None,
    state=None,
    top_k=50
):
    """
    PROGRAM-LEVEL semantic ranking.
    Clean cosine similarity + geographic filtering + top_units from chunk index.
    """
    query_vector = model.encode([interest])
    faiss.normalize_L2(query_vector)
    similarities, indices = index.search(query_vector, top_k)

    registry = load_registry()

    # Build allowed program set based on geographic filter
    allowed_programs = set()
    for entry in registry:
        if not entry.get("is_active", False):
            continue
        if country and normalize(entry.get("country")) != normalize(country):
            continue
        if state and normalize(entry.get("state")) != normalize(state):
            continue
        key = (normalize(entry.get("college")), normalize(entry.get("program")))
        allowed_programs.add(key)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        item = metadata[idx]
        key = (normalize(item["college"]), normalize(item["program"]))

        if country or state:
            if key not in allowed_programs:
                continue

        similarity = float(similarities[0][i])

        # Get top matching chunks for this program
        top_units = get_top_units(query_vector, item["college"], item["program"], top_k=3)

        results.append({
            "college": item["college"],
            "program": item["program"],
            "score": round(similarity, 4),
            "alignment_strength": classify_alignment(similarity),
            "explainability": {
                "average_similarity": round(similarity, 4),
                "matched_unit_count": len(top_units),
                "coverage_factor": 1.0,
                "alignment_strength": classify_alignment(similarity)
            },
            "syllabus_pdf": item.get("file_path", "N/A"),
            "top_units": top_units
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)
