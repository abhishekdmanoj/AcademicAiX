import faiss
import json
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UNIVERSITY_METADATA_PATH = os.path.join(PROJECT_ROOT, "data", "university_metadata.json")
REGISTRY_PATH = os.path.join(PROJECT_ROOT, "data", "registry.json")


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


def rank_universities(
    interest,
    model,
    index,
    metadata,
    country=None,
    state=None,
    top_k=50  # increased to allow post-filtering safely
):
    """
    PROGRAM-LEVEL semantic ranking.
    Clean cosine similarity + geographic filtering.
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

        # If geographic filter exists → enforce it
        if country or state:
            if key not in allowed_programs:
                continue

        similarity = float(similarities[0][i])

        results.append({
            "college": item["college"],
            "program": item["program"],
            "score": round(similarity, 4),
            "explainability": {
                "average_similarity": round(similarity, 4),
                "matched_unit_count": 1,
                "coverage_factor": 1.0,
                "alignment_strength": classify_alignment(similarity)
            },
            "syllabus_pdf": item.get("file_path", "N/A"),
            "top_units": []
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)