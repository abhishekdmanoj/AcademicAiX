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
    """
    Alignment classification based on cosine similarity.
    Calibrated for small corpus.
    """
    if score >= 0.55:
        return "Strong"
    elif score >= 0.40:
        return "Moderate"
    else:
        return "Weak"


def rank_universities(interest, model, index, metadata, top_k=50):
    """
    Returns ranked programs with explainability.
    Uses Top-3 strongest unit similarities.
    Suppresses weak semantic noise using threshold.
    Always returns consistent schema (list of program objects).
    """

    # Encode user query
    query_vector = model.encode([interest])
    faiss.normalize_L2(query_vector)

    # Search FAISS
    similarities, indices = index.search(query_vector, top_k)

    university_scores = {}
    university_units = {}
    university_pdf_paths = {}

    # Collect matches
    for i, idx in enumerate(indices[0]):

        if idx == -1:
            continue

        item = metadata[idx]
        college = item["college"]
        program = item["program"]
        program_key = f"{college}||{program}"

        similarity = float(similarities[0][i])

        university_scores.setdefault(program_key, []).append(similarity)
        university_units.setdefault(program_key, []).append(
            (item["unit"], similarity)
        )
        university_pdf_paths[program_key] = item.get("file_path", "N/A")

    results = []

    # 🔥 Adjusted threshold (calibrated)
    SIMILARITY_THRESHOLD = 0.28

    # Aggregate per program
    for program_key, sim_list in university_scores.items():

        # Ignore weak semantic noise
        strong_sims = [s for s in sim_list if s >= SIMILARITY_THRESHOLD]

        if not strong_sims:
            continue

        # Top-3 strongest similarities define specialization
        top_sims = sorted(strong_sims, reverse=True)[:3]

        mean_similarity = sum(top_sims) / len(top_sims)
        final_score = mean_similarity

        # Top 3 explainability units
        top_units = sorted(
            [u for u in university_units[program_key] if u[1] >= SIMILARITY_THRESHOLD],
            key=lambda x: x[1],
            reverse=True
        )[:3]

        college, program = program_key.split("||")

        results.append({
            "college": college,
            "program": program,
            "score": round(final_score, 4),
            "explainability": {
                "average_similarity": round(mean_similarity, 4),
                "matched_unit_count": len(top_sims),
                "alignment_strength": classify_alignment(final_score)
            },
            "syllabus_pdf": university_pdf_paths.get(program_key, "N/A"),
            "top_units": [
                {
                    "unit": unit_text,
                    "similarity": round(sim, 4)
                }
                for unit_text, sim in top_units
            ]
        })

    # Always return consistent schema
    if not results:
        return []

    return sorted(results, key=lambda x: x["score"], reverse=True)