import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.config import UPLOADS_DIR, settings
from app.models.schemas import PreprocessRequest, PreprocessResult, Job, JobStatus
from app.services.job_manager import JobManager
from app.services.preprocessor import SketchPreprocessor

router = APIRouter(prefix="/api", tags=["Preprocessing"])

@router.post("/upload", response_model=Job)
async def upload_sketch(file: UploadFile = File(...)):
    if file.content_type not in settings.allowed_mime_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {file.content_type}. Allowed: {settings.allowed_mime_types}"
        )

    job_id = f"job_{uuid.uuid4().hex[:10]}"
    extension = Path(file.filename or "sketch.png").suffix or ".png"
    dest_path = UPLOADS_DIR / f"{job_id}_raw{extension}"

    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    try:
        SketchPreprocessor.validate_image_file(dest_path, max_size_bytes=settings.max_upload_size_bytes)
    except Exception as e:
        if dest_path.exists():
            dest_path.unlink()
        raise HTTPException(status_code=400, detail=str(e))

    url = f"/storage/uploads/{dest_path.name}"
    job = JobManager.create_job(job_id=job_id, original_path=str(dest_path.resolve()), original_url=url)
    return job

@router.post("/preprocess/{job_id}", response_model=PreprocessResult)
async def preprocess_sketch(
    job_id: str,
    request: PreprocessRequest
):
    job = JobManager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if not job.original_image_path or not Path(job.original_image_path).exists():
        raise HTTPException(status_code=400, detail="Original image not found for job")

    try:
        result = SketchPreprocessor.process_sketch(
            image_path=Path(job.original_image_path),
            job_id=job_id,
            params=request
        )
        edge_path = Path(result.preprocessed_url.replace("/storage/", str(UPLOADS_DIR.parent) + "/"))
        job.preprocessed_image_path = str(edge_path.resolve())
        job.preprocessed_image_url = result.preprocessed_url
        job.preprocess_config = request.model_dump()
        job.status = JobStatus.PREPROCESSED
        JobManager.save_job(job)
        return result
    except Exception as e:
        JobManager.update_job_status(job_id, JobStatus.FAILED, error_message=str(e))
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {e}")
