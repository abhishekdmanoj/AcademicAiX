import faiss
import json
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UNIVERSITY_METADATA_PATH = os.path.join(PROJECT_ROOT, "data", "university_metadata.json")


def load_university_metadata():
    if not os.path.exists(UNIVERSITY_METADATA_PATH):
        return {}
    with open(UNIVERSITY_METADATA_PATH, "r") as f:
        return json.load(f)


def classify_alignment(score):
    if score >= 0.30:
        return "Strong"
    elif score >= 0.20:
        return "Moderate"
    else:
        return "Weak"


def rank_universities(interest, model, index, metadata, top_k=10):
    """
    PROGRAM-LEVEL semantic ranking.
    Clean cosine similarity.
    """

    query_vector = model.encode([interest])
    faiss.normalize_L2(query_vector)

    similarities, indices = index.search(query_vector, top_k)

    results = []

    for i, idx in enumerate(indices[0]):

        if idx == -1:
            continue

        item = metadata[idx]

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