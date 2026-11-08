import shutil
from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from app.config import STORAGE_DIR
from app.models.schemas import Job, JobStatus
from app.services.job_manager import JobManager

router = APIRouter(prefix="/api/jobs", tags=["Jobs & Transfer"])

@router.get("", response_model=List[Job])
async def list_all_jobs(limit: int = 50):
    return JobManager.list_jobs(limit=limit)

@router.get("/{job_id}", response_model=Job)
async def get_single_job(job_id: str):
    job = JobManager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job

@router.get("/{job_id}/export-bundle")
async def export_job_bundle(job_id: str):
    job = JobManager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if not job.preprocessed_image_path:
        raise HTTPException(status_code=400, detail="Job has not been preprocessed yet")

    try:
        bundle_path = JobManager.export_job_bundle(job_id)
        job.status = JobStatus.PENDING_GPU
        JobManager.save_job(job)
        return FileResponse(
            path=bundle_path,
            filename=bundle_path.name,
            media_type="application/zip"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create Kaggle bundle: {e}")

@router.post("/{job_id}/import-result", response_model=Job)
async def import_job_result(job_id: str, file: UploadFile = File(...)):
    job = JobManager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    temp_zip = STORAGE_DIR / f"temp_{job_id}_{file.filename}"
    try:
        with open(temp_zip, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        updated_job = JobManager.import_job_result(job_id, temp_zip)
        return updated_job
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to import result bundle: {e}")
    finally:
        if temp_zip.exists():
            temp_zip.unlink()
