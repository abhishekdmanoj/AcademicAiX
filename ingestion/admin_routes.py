import os
import json
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

from ingestion.auto_ingest import (
    extract_metadata,
    register_program,
    infer_degree_level,
    load_json,
    save_json,
    REGISTRY_PATH,
    ALLOWED_UPLOAD_FOLDER
)

router = APIRouter(prefix="/admin", tags=["Admin"])

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METADATA_PATH = os.path.join(PROJECT_ROOT, "data", "university_metadata.json")


def is_safe_path(folder_path):
    real_path = os.path.realpath(folder_path)
    allowed_real = os.path.realpath(ALLOWED_UPLOAD_FOLDER)
    return real_path.startswith(allowed_real)


# ----------------------------
# POST /admin/ingest-pdf
# ----------------------------

@router.post("/ingest-pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        metadata = extract_metadata(tmp_path)
        return JSONResponse({"success": True, "tmp_path": tmp_path, "metadata": metadata})
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# POST /admin/confirm-ingest
# ----------------------------

@router.post("/confirm-ingest")
async def confirm_ingest(
    tmp_path: str = Form(...),
    college: str = Form(...),
    program: str = Form(...),
    degree_level: str = Form(...),
    country: str = Form("India"),
    state: str = Form(""),
    source_url: str = Form("")
):
    if not os.path.exists(tmp_path):
        raise HTTPException(status_code=400, detail="Temp file not found. Please re-upload.")

    result = register_program(
        pdf_path=tmp_path,
        college=college,
        program=program,
        degree_level=degree_level,
        country=country,
        state=state,
        source_url=source_url
    )

    if not result["success"]:
        return JSONResponse({"success": False, "message": result["message"]})

    try:
        from offline_pipeline.build_syllabus_index import build_syllabus_index
        build_syllabus_index()
    except Exception as e:
        return JSONResponse({"success": False, "message": f"Registered but rebuild failed: {str(e)}"})

    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

    return JSONResponse({"success": True, "message": result["message"]})


# ----------------------------
# POST /admin/ingest-folder
# ----------------------------

@router.post("/ingest-folder")
async def ingest_folder(
    background_tasks: BackgroundTasks,
    folder_path: str = Form(...),
    country: str = Form("India"),
    state: str = Form("")
):
    if not is_safe_path(folder_path):
        raise HTTPException(status_code=403, detail=f"Folder must be inside: {ALLOWED_UPLOAD_FOLDER}")

    if not os.path.exists(folder_path):
        raise HTTPException(status_code=400, detail=f"Folder not found: {folder_path}")

    pdfs = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdfs:
        return JSONResponse({"success": False, "message": "No PDFs found in folder"})

    results = []
    success_count = 0

    for pdf_file in pdfs:
        full_path = os.path.join(folder_path, pdf_file)
        try:
            metadata = extract_metadata(full_path)
            if not metadata["college"] or not metadata["program"]:
                results.append({"file": pdf_file, "success": False, "message": "Could not detect metadata"})
                continue

            result = register_program(
                pdf_path=full_path,
                college=metadata["college"],
                program=metadata["program"],
                degree_level=metadata["degree_level"],
                country=country,
                state=state
            )

            results.append({
                "file": pdf_file,
                "success": result["success"],
                "message": result["message"],
                "detected": {"college": metadata["college"], "program": metadata["program"]}
            })

            if result["success"]:
                success_count += 1

        except Exception as e:
            results.append({"file": pdf_file, "success": False, "message": str(e)})

    if success_count > 0:
        try:
            from offline_pipeline.build_syllabus_index import build_syllabus_index
            build_syllabus_index()
        except Exception as e:
            return JSONResponse({
                "success": False,
                "message": f"Ingested {success_count} but rebuild failed: {str(e)}",
                "results": results
            })

    return JSONResponse({"success": True, "ingested": success_count, "total": len(pdfs), "results": results})


# ----------------------------
# POST /admin/check-updates
# ----------------------------

@router.post("/check-updates")
async def check_updates(background_tasks: BackgroundTasks):
    try:
        from ingestion.check_for_updates import main as run_check
        background_tasks.add_task(run_check)
        return JSONResponse({"success": True, "message": "Update check started in background."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# GET /admin/programs
# ----------------------------

@router.get("/programs")
async def list_programs():
    registry = load_json(REGISTRY_PATH)
    return JSONResponse({"programs": registry})


# ----------------------------
# PATCH /admin/program/toggle
# ----------------------------

@router.patch("/program/toggle")
async def toggle_program(
    college: str = Form(...),
    program: str = Form(...)
):
    registry = load_json(REGISTRY_PATH)

    for entry in registry:
        if entry.get("college") == college and entry.get("program") == program:
            entry["is_active"] = not entry.get("is_active", True)
            save_json(REGISTRY_PATH, registry)
            status = "activated" if entry["is_active"] else "deactivated"
            return JSONResponse({"success": True, "message": f"{college} - {program} {status}"})

    raise HTTPException(status_code=404, detail="Program not found")


# ----------------------------
# POST /admin/bulk-scrape
# ----------------------------

@router.post("/bulk-scrape")
async def bulk_scrape(background_tasks: BackgroundTasks):
    try:
        from ingestion.pdf_downloader import bulk_download
        background_tasks.add_task(bulk_download)
        return JSONResponse({"success": True, "message": "Bulk scrape started. Check server logs for progress."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# GET /admin/program/metadata
# ----------------------------

@router.get("/program/metadata")
async def get_program_metadata(college: str, program: str):
    """Get metadata for a specific program"""
    metadata = load_json(METADATA_PATH)

    for entry in metadata:
        if (entry.get("college", "").lower() == college.lower() and
                entry.get("program", "").lower() == program.lower()):
            return JSONResponse({"success": True, "metadata": entry})

    # Return empty template if not found
    return JSONResponse({
        "success": True,
        "metadata": {
            "college": college,
            "program": program,
            "official_website": "",
            "entrance_exams": [],
            "pyq_links": []
        }
    })


# ----------------------------
# POST /admin/program/metadata
# ----------------------------

class EntranceExam(BaseModel):
    name: str
    website: Optional[str] = ""
    syllabus_pdf: Optional[str] = ""


class ProgramMetadataUpdate(BaseModel):
    college: str
    program: str
    official_website: Optional[str] = ""
    entrance_exams: Optional[List[EntranceExam]] = []
    pyq_links: Optional[List[str]] = []


@router.post("/program/metadata")
async def update_program_metadata(req: ProgramMetadataUpdate):
    """Create or update metadata for a program"""
    metadata = load_json(METADATA_PATH)

    # Find existing entry
    found = False
    for entry in metadata:
        if (entry.get("college", "").lower() == req.college.lower() and
                entry.get("program", "").lower() == req.program.lower()):
            entry["official_website"] = req.official_website
            entry["entrance_exams"] = [e.dict() for e in req.entrance_exams]
            entry["pyq_links"] = req.pyq_links
            found = True
            break

    # Create new entry if not found
    if not found:
        metadata.append({
            "college": req.college,
            "program": req.program,
            "official_website": req.official_website,
            "entrance_exams": [e.dict() for e in req.entrance_exams],
            "pyq_links": req.pyq_links,
            "last_updated": str(__import__("datetime").date.today())
        })

    # Save
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    return JSONResponse({
        "success": True,
        "message": f"Metadata saved for {req.college} - {req.program}"
    })


# ----------------------------
# POST /admin/program/registry
# ----------------------------

class RegistryUpdate(BaseModel):
    original_college: str
    original_program: str
    college: str
    program: str
    degree_level: Optional[str] = "UG"
    country: Optional[str] = "India"
    state: Optional[str] = ""
    source_url: Optional[str] = ""


@router.post("/program/registry")
async def update_program_registry(req: RegistryUpdate):
    """Edit registry fields for an existing program"""
    registry = load_json(REGISTRY_PATH)

    found = False
    for entry in registry:
        if (entry.get("college", "").lower() == req.original_college.lower() and
                entry.get("program", "").lower() == req.original_program.lower()):
            entry["college"] = req.college
            entry["program"] = req.program
            entry["degree_level"] = req.degree_level
            entry["country"] = req.country
            entry["state"] = req.state
            entry["source_url"] = req.source_url
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail="Program not found in registry")

    save_json(REGISTRY_PATH, registry)

    # If college or program name changed, update metadata.json too
    if req.original_college != req.college or req.original_program != req.program:
        metadata = load_json(METADATA_PATH)
        for entry in metadata:
            if (entry.get("college", "").lower() == req.original_college.lower() and
                    entry.get("program", "").lower() == req.original_program.lower()):
                entry["college"] = req.college
                entry["program"] = req.program
                break
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)

    return JSONResponse({
        "success": True,
        "message": f"Registry updated for {req.college} - {req.program}"
    })


# ----------------------------
# GET /admin/sources
# ----------------------------

SOURCES_PATH = os.path.join(PROJECT_ROOT, "data", "sources.json")

@router.get("/sources")
async def get_sources():
    sources = load_json(SOURCES_PATH)
    return JSONResponse({"success": True, "sources": sources})


# ----------------------------
# POST /admin/sources
# ----------------------------

class SourceEntry(BaseModel):
    college: str
    country: Optional[str] = "India"
    state: Optional[str] = ""
    type: Optional[str] = "direct_pdf"
    urls: List[str] = []

class SourceUpdate(BaseModel):
    original_college: Optional[str] = None
    source: SourceEntry

@router.post("/sources")
async def upsert_source(req: SourceUpdate):
    sources = load_json(SOURCES_PATH)

    if req.original_college:
        # Update existing
        found = False
        for entry in sources:
            if entry.get("college", "").lower() == req.original_college.lower():
                entry["college"] = req.source.college
                entry["country"] = req.source.country
                entry["state"] = req.source.state
                entry["type"] = req.source.type
                entry["urls"] = req.source.urls
                found = True
                break
        if not found:
            raise HTTPException(status_code=404, detail="Source not found")
        msg = f"Source updated for {req.source.college}"
    else:
        # Add new
        for entry in sources:
            if entry.get("college", "").lower() == req.source.college.lower():
                raise HTTPException(status_code=400, detail=f"{req.source.college} already exists in sources")
        sources.append({
            "college": req.source.college,
            "country": req.source.country,
            "state": req.source.state,
            "type": req.source.type,
            "urls": req.source.urls
        })
        msg = f"Source added for {req.source.college}"

    with open(SOURCES_PATH, "w") as f:
        json.dump(sources, f, indent=2)

    return JSONResponse({"success": True, "message": msg})


# ----------------------------
# DELETE /admin/sources/{college}
# ----------------------------

@router.delete("/sources/{college}")
async def delete_source(college: str):
    sources = load_json(SOURCES_PATH)
    original_len = len(sources)
    sources = [s for s in sources if s.get("college", "").lower() != college.lower()]

    if len(sources) == original_len:
        raise HTTPException(status_code=404, detail="Source not found")

    with open(SOURCES_PATH, "w") as f:
        json.dump(sources, f, indent=2)

    return JSONResponse({"success": True, "message": f"Source removed: {college}"})
