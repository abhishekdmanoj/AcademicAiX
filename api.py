import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from embeddings.model import load_embedding_model
from runtime.index_loader import load_syllabus_index
from runtime.ranking_service import rank_universities


app = FastAPI(title="AcademicAiX")

# ----------------------------
# Static File Mount (Serve PDFs)
# ----------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")

app.mount("/data", StaticFiles(directory=DATA_PATH), name="data")

# ----------------------------
# CORS Configuration
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# Utility Normalization
# ----------------------------
def normalize(text):
    if not text:
        return ""
    return text.strip().lower()


# ----------------------------
# Request Models
# ----------------------------
class InterestRequest(BaseModel):
    interest: str


class ProgramRequest(BaseModel):
    college: str
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

    cleaned = []

    for r in results:
        cleaned.append({
            "college": r["college"],
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

    # ----------------------------
    # Load Registry (for syllabus path)
    # ----------------------------
    registry_path = os.path.join(PROJECT_ROOT, "data", "registry.json")

    if not os.path.exists(registry_path):
        return {"error": "Registry not found."}

    with open(registry_path, "r") as f:
        registry = json.load(f)

    syllabus_path = None

    for entry in registry:
        if (
            normalize(entry.get("college")) == normalize(req.college)
            and normalize(entry.get("program")) == normalize(req.program)
            and entry.get("is_active", False)
        ):
            syllabus_path = entry.get("file_path")
            break

    # ----------------------------
    # Load University Metadata (New Structure)
    # ----------------------------
    metadata_path = os.path.join(PROJECT_ROOT, "data", "university_metadata.json")

    if not os.path.exists(metadata_path):
        return {"error": "Metadata not found."}

    with open(metadata_path, "r") as f:
        metadata_list = json.load(f)

    info = None

    for entry in metadata_list:
        if (
            normalize(entry.get("college")) == normalize(req.college)
            and normalize(entry.get("program")) == normalize(req.program)
        ):
            info = entry
            break

    if not info:
        return {"error": "Program not found in metadata."}

    # ----------------------------
    # Unified Response (NEW STRUCTURE)
    # ----------------------------
    return {
        "college": req.college,
        "program": req.program,
        "syllabus_pdf": syllabus_path,
        "official_website": info.get("official_website"),
        "entrance_exams": info.get("entrance_exams", []),
        "pyq_links": info.get("pyq_links", [])
    }