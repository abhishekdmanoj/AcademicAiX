import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

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


def is_safe_path(folder_path):
    """Ensure folder is within allowed upload directory"""
    real_path = os.path.realpath(folder_path)
    allowed_real = os.path.realpath(ALLOWED_UPLOAD_FOLDER)
    return real_path.startswith(allowed_real)


# ─────────────────────────────────────────
# POST /admin/ingest-pdf
# ─────────────────────────────────────────

@router.post("/ingest-pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    """Upload PDF → extract metadata → return for admin confirmation"""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files accepted"
        )

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf"
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        metadata = extract_metadata(tmp_path)
        return JSONResponse({
            "success": True,
            "tmp_path": tmp_path,
            "metadata": metadata
        })
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# POST /admin/confirm-ingest
# ─────────────────────────────────────────

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
    """Admin confirms metadata → register → rebuild index"""
    if not os.path.exists(tmp_path):
        raise HTTPException(
            status_code=400,
            detail="Temp file not found. Please re-upload."
        )

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
        return JSONResponse({
            "success": False,
            "message": result["message"]
        })

    try:
        from offline_pipeline.build_syllabus_index import build_syllabus_index
        build_syllabus_index()
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Registered but rebuild failed: {str(e)}"
        })

    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

    return JSONResponse({
        "success": True,
        "message": result["message"]
    })


# ─────────────────────────────────────────
# POST /admin/ingest-folder
# ─────────────────────────────────────────

@router.post("/ingest-folder")
async def ingest_folder(
    background_tasks: BackgroundTasks,
    folder_path: str = Form(...),
    country: str = Form("India"),
    state: str = Form("")
):
    """
    Ingest all PDFs from a folder.
    Restricted to ALLOWED_UPLOAD_FOLDER only.
    Rebuilds index ONCE after all ingestions.
    """
    if not is_safe_path(folder_path):
        raise HTTPException(
            status_code=403,
            detail=f"Folder must be inside: {ALLOWED_UPLOAD_FOLDER}"
        )

    if not os.path.exists(folder_path):
        raise HTTPException(
            status_code=400,
            detail=f"Folder not found: {folder_path}"
        )

    pdfs = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdfs:
        return JSONResponse({
            "success": False,
            "message": "No PDFs found in folder"
        })

    results = []
    success_count = 0

    for pdf_file in pdfs:
        full_path = os.path.join(folder_path, pdf_file)
        try:
            metadata = extract_metadata(full_path)

            if not metadata["college"] or not metadata["program"]:
                results.append({
                    "file": pdf_file,
                    "success": False,
                    "message": "Could not detect metadata"
                })
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
                "detected": {
                    "college": metadata["college"],
                    "program": metadata["program"]
                }
            })

            if result["success"]:
                success_count += 1

        except Exception as e:
            results.append({
                "file": pdf_file,
                "success": False,
                "message": str(e)
            })

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

    return JSONResponse({
        "success": True,
        "ingested": success_count,
        "total": len(pdfs),
        "results": results
    })


# ─────────────────────────────────────────
# POST /admin/check-updates
# ─────────────────────────────────────────

@router.post("/check-updates")
async def check_updates(background_tasks: BackgroundTasks):
    """Triggers check_for_updates pipeline in background"""
    try:
        from ingestion.check_for_updates import main as run_check
        background_tasks.add_task(run_check)
        return JSONResponse({
            "success": True,
            "message": "Update check started in background. Check server logs for results."
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# GET /admin/programs
# ─────────────────────────────────────────

@router.get("/programs")
async def list_programs():
    """Returns all programs in registry"""
    registry = load_json(REGISTRY_PATH)
    return JSONResponse({"programs": registry})


# ─────────────────────────────────────────
# PATCH /admin/program/toggle
# ─────────────────────────────────────────

@router.patch("/program/toggle")
async def toggle_program(
    college: str = Form(...),
    program: str = Form(...)
):
    """Toggle is_active flag for a program"""
    registry = load_json(REGISTRY_PATH)

    for entry in registry:
        if (
            entry.get("college") == college and
            entry.get("program") == program
        ):
            entry["is_active"] = not entry.get("is_active", True)
            save_json(REGISTRY_PATH, registry)
            status = "activated" if entry["is_active"] else "deactivated"
            return JSONResponse({
                "success": True,
                "message": f"{college} - {program} {status}"
            })

    raise HTTPException(status_code=404, detail="Program not found")


@router.post("/bulk-scrape")
async def bulk_scrape(background_tasks: BackgroundTasks):
    """Trigger bulk download from all sources in sources.json"""
    try:
        from ingestion.pdf_downloader import bulk_download
        background_tasks.add_task(bulk_download)
        return JSONResponse({
            "success": True,
            "message": "Bulk scrape started. Check server logs for progress."
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))