from pathlib import Path
from fastapi import APIRouter, HTTPException
from app.models.schemas import InferenceRequest, Job, JobStatus
from app.services.job_manager import JobManager
from app.services.inference_engine import InferenceEngine

router = APIRouter(prefix="/api/inference", tags=["Inference"])

@router.post("/{job_id}", response_model=Job)
async def run_job_inference(job_id: str, request: InferenceRequest):
    job = JobManager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if not job.preprocessed_image_path or not Path(job.preprocessed_image_path).exists():
        raise HTTPException(status_code=400, detail="Sketch must be preprocessed before running inference")

    try:
        job.prompt_config = request.prompt_options.model_dump()
        result_path, metadata = InferenceEngine.run_inference(
            edge_image_path=Path(job.preprocessed_image_path),
            job_id=job_id,
            request=request
        )

        job.result_image_path = str(result_path.resolve())
        job.result_image_url = f"/storage/results/{result_path.name}"
        job.generation_metadata = metadata
        job.status = JobStatus.COMPLETED
        JobManager.save_job(job)
        return job
    except Exception as e:
        JobManager.update_job_status(job_id, JobStatus.FAILED, error_message=str(e))
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {e}")
