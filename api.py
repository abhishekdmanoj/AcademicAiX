import os
import json
from fastapi import FastAPI
from pydantic import BaseModel

from embeddings.model import load_embedding_model
from runtime.index_loader import load_syllabus_index
from runtime.ranking_service import rank_universities


app = FastAPI(title="AcademicAiX")

# ----------------------------
# Global Runtime Objects
# ----------------------------
model = None
syll_index = None
syll_meta = None


@app.on_event("startup")
def startup():
    global model, syll_index, syll_meta
    print("Loading embedding model...")
    model = load_embedding_model()
    print("Loading FAISS index...")
    syll_index, syll_meta = load_syllabus_index()
    print("Startup complete.")


# ----------------------------
# Request Models
# ----------------------------
class InterestRequest(BaseModel):
    interest: str


class ProgramRequest(BaseModel):
    program: str


# ----------------------------
# Rank Endpoint
# ----------------------------
@app.post("/rank")
def rank(req: InterestRequest):

    results = rank_universities(
        req.interest,
        model,
        syll_index,
        syll_meta
    )

    # Clean response for frontend
    cleaned = []

    for r in results:
        cleaned.append({
            "program": r["program"],
            "score": r["score"],
            "alignment_strength": r["explainability"]["alignment_strength"],
            "top_units": r["top_units"]
        })

    return {"results": cleaned}


# ----------------------------
# Program Details Endpoint
# ----------------------------
@app.post("/program-details")
def program_details(req: ProgramRequest):

    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

    # ----------------------------
    # Load Registry (for syllabus)
    # ----------------------------
    registry_path = os.path.join(PROJECT_ROOT, "data", "registry.json")

    if not os.path.exists(registry_path):
        return {"error": "Registry not found."}

    with open(registry_path, "r") as f:
        registry = json.load(f)

    syllabus_path = None

    for entry in registry:
        if (
            entry["university"] == req.program
            and entry.get("is_active", False)
        ):
            syllabus_path = entry.get("file_path")
            break

    # ----------------------------
    # Load University Metadata
    # ----------------------------
    metadata_path = os.path.join(PROJECT_ROOT, "data", "university_metadata.json")

    if not os.path.exists(metadata_path):
        return {"error": "Metadata not found."}

    with open(metadata_path, "r") as f:
        data = json.load(f)

    info = data.get(req.program)

    if not info:
        return {"error": "Program not found."}

    # ----------------------------
    # Unified Response
    # ----------------------------
    return {
        "program": req.program,
        "syllabus_pdf": syllabus_path,
        "entrance": info.get("entrance"),
        "entrance_website": info.get("entrance_website"),
        "pyq_links": info.get("pyq_links", [])
    }