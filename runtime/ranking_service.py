import faiss
import math
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
    if score >= 0.45:
        return "Strong"
    elif score >= 0.30:
        return "Moderate"
    else:
        return "Weak"


def rank_universities(interest, model, index, metadata, top_k=50):
    """
    Returns ranked programs with explainability.
    Does NOT return entrance/PYQ info.
    """

    query_vector = model.encode([interest])
    faiss.normalize_L2(query_vector)

    similarities, indices = index.search(query_vector, top_k)

    university_scores = {}
    university_units = {}
    university_pdf_paths = {}

    for i, idx in enumerate(indices[0]):
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

    for program_key, sim_list in university_scores.items():

        positive_sims = [s for s in sim_list if s > 0]

        if not positive_sims:
            continue

        top_sims = sorted(positive_sims, reverse=True)[:5]

        mean_similarity = sum(top_sims) / len(top_sims)
        coverage_factor = 1 + (len(top_sims) / 5)
        final_score = mean_similarity * coverage_factor

        top_units = sorted(
            [u for u in university_units[program_key] if u[1] > 0],
            key=lambda x: x[1],
            reverse=True
        )[:5]

        college, program = program_key.split("||")

        results.append({
            "college": college,
            "program": program,
            "score": round(final_score, 4),
            "explainability": {
                "average_similarity": round(mean_similarity, 4),
                "matched_unit_count": len(top_sims),
                "coverage_factor": round(coverage_factor, 2),
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

    if not results:
        return [{"message": "No strong matches found."}]

    return sorted(results, key=lambda x: x["score"], reverse=True)