from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from app.models.schemas import EvaluationMetrics, Job
from app.services.job_manager import JobManager
from app.services.evaluator import SceneEvaluator

router = APIRouter(prefix="/api/eval", tags=["Evaluation"])

@router.post("/{job_id}", response_model=EvaluationMetrics)
async def evaluate_job(job_id: str, tolerance: int = Query(default=2, ge=0, le=10)):
    job = JobManager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if not job.preprocessed_image_path or not Path(job.preprocessed_image_path).exists():
        raise HTTPException(status_code=400, detail="Preprocessed sketch edge image is required for evaluation")

    if not job.result_image_path or not Path(job.result_image_path).exists():
        raise HTTPException(status_code=400, detail="Result render image is required for evaluation")

    try:
        metrics = SceneEvaluator.evaluate_render(
            sketch_edge_path=Path(job.preprocessed_image_path),
            result_image_path=Path(job.result_image_path),
            tolerance_radius=tolerance
        )
        job.evaluation = metrics
        JobManager.save_job(job)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")
