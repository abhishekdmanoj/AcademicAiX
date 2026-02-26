import faiss
import json
import os
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UNIVERSITY_METADATA_PATH = os.path.join(PROJECT_ROOT, "data", "university_metadata.json")


def classify_alignment(score):
    """
    Alignment classification based on normalized score.
    """
    if score >= 0.75:
        return "Strong"
    elif score >= 0.45:
        return "Moderate"
    else:
        return "Weak"


def rank_universities(interest, model, index, metadata, top_k=50):
    """
    Final academic ranking implementation.
    Weighted semantic aggregation.
    No hard thresholds.
    Query-normalized.
    """

    # Encode query
    query_vector = model.encode([interest])
    faiss.normalize_L2(query_vector)

    similarities, indices = index.search(query_vector, top_k)

    program_sims = {}
    program_units = {}
    program_pdf_paths = {}

    # Collect similarities per program
    for i, idx in enumerate(indices[0]):

        if idx == -1:
            continue

        item = metadata[idx]
        college = item["college"]
        program = item["program"]
        key = f"{college}||{program}"

        sim = float(similarities[0][i])

        program_sims.setdefault(key, []).append(sim)
        program_units.setdefault(key, []).append((item["unit"], sim))
        program_pdf_paths[key] = item.get("file_path", "N/A")

    raw_scores = {}
    explain_data = {}

    for key, sims in program_sims.items():

        # Take top 5 similarities
        top_sims = sorted(sims, reverse=True)[:5]

        if len(top_sims) == 0:
            continue

        sims_array = np.array(top_sims)

        # Weighted specialization score
        weighted_score = np.sum(sims_array ** 2) / np.sum(sims_array)

        # Peak reinforcement
        peak = np.max(sims_array)

        # Combined raw score
        raw_score = 0.7 * weighted_score + 0.3 * peak

        raw_scores[key] = raw_score
        explain_data[key] = {
            "top_sims": top_sims,
            "weighted_score": weighted_score,
            "peak": peak
        }

    if not raw_scores:
        return []

    # Normalize scores per query
    min_score = min(raw_scores.values())
    max_score = max(raw_scores.values())

    results = []

    for key, raw in raw_scores.items():

        if max_score == min_score:
            normalized = 0.5
        else:
            normalized = (raw - min_score) / (max_score - min_score)

        college, program = key.split("||")

        top_units = sorted(
            program_units[key],
            key=lambda x: x[1],
            reverse=True
        )[:3]

        results.append({
            "college": college,
            "program": program,
            "score": round(normalized, 4),
            "explainability": {
                "weighted_semantic_score": round(explain_data[key]["weighted_score"], 4),
                "peak_similarity": round(explain_data[key]["peak"], 4),
                "alignment_strength": classify_alignment(normalized)
            },
            "syllabus_pdf": program_pdf_paths.get(key, "N/A"),
            "top_units": [
                {
                    "unit": unit_text,
                    "similarity": round(sim, 4)
                }
                for unit_text, sim in top_units
            ]
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)