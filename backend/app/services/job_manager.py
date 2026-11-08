import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
from app.config import JOBS_DIR, BUNDLES_DIR, RESULTS_DIR
from app.models.schemas import Job, JobStatus, GenerationMetadata, EvaluationMetrics

class JobManager:
    @staticmethod
    def _get_job_path(job_id: str) -> Path:
        return JOBS_DIR / f"{job_id}.json"

    @classmethod
    def create_job(cls, job_id: str, original_path: Optional[str] = None, original_url: Optional[str] = None) -> Job:
        now = datetime.now(timezone.utc).isoformat()
        job = Job(
            id=job_id,
            status=JobStatus.CREATED,
            created_at=now,
            updated_at=now,
            original_image_path=original_path,
            original_image_url=original_url
        )
        cls.save_job(job)
        return job

    @classmethod
    def save_job(cls, job: Job) -> None:
        job.updated_at = datetime.now(timezone.utc).isoformat()
        path = cls._get_job_path(job.id)
        with open(path, "w", encoding="utf-8") as f:
            f.write(job.model_dump_json(indent=2))

    @classmethod
    def get_job(cls, job_id: str) -> Optional[Job]:
        path = cls._get_job_path(job_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Job.model_validate(data)

    @classmethod
    def list_jobs(cls, limit: int = 50) -> List[Job]:
        jobs = []
        files = sorted(JOBS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[:limit]:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    jobs.append(Job.model_validate(json.load(f)))
            except Exception:
                continue
        return jobs

    @classmethod
    def update_job_status(cls, job_id: str, status: JobStatus, error_message: Optional[str] = None) -> Optional[Job]:
        job = cls.get_job(job_id)
        if not job:
            return None
        job.status = status
        if error_message:
            job.error_message = error_message
        cls.save_job(job)
        return job

    @classmethod
    def export_job_bundle(cls, job_id: str) -> Path:
        job = cls.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        if not job.preprocessed_image_path or not Path(job.preprocessed_image_path).exists():
            raise ValueError("Preprocessed sketch not found for job")

        bundle_path = BUNDLES_DIR / f"job_{job_id}_kaggle_bundle.zip"
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add preprocessed edge image
            edge_path = Path(job.preprocessed_image_path)
            zf.write(edge_path, arcname="control_sketch.png")
            
            # Add job config and instructions
            bundle_manifest = {
                "job_id": job.id,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "prompt_config": job.prompt_config or {},
                "preprocess_config": job.preprocess_config or {},
                "instructions": (
                    "Upload this bundle or extract control_sketch.png and pass prompt_config to "
                    "the Kaggle notebook inference pipeline."
                )
            }
            zf.writestr("manifest.json", json.dumps(bundle_manifest, indent=2))
        return bundle_path

    @classmethod
    def import_job_result(cls, job_id: str, zip_path: Path) -> Job:
        job = cls.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        extract_dir = RESULTS_DIR / job_id
        extract_dir.mkdir(parents=True, exist_ok=True)

        result_image_dest = RESULTS_DIR / f"{job_id}_result.png"
        metadata_dict: Dict[str, Any] = {}

        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                lower_name = name.lower()
                if lower_name.endswith((".png", ".jpg", ".jpeg", ".webp")) and "result" in lower_name:
                    with zf.open(name) as src, open(result_image_dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                elif lower_name.endswith((".png", ".jpg", ".jpeg", ".webp")) and not result_image_dest.exists():
                    with zf.open(name) as src, open(result_image_dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                elif lower_name.endswith(".json") and ("meta" in lower_name or "manifest" in lower_name):
                    raw = zf.read(name).decode("utf-8")
                    try:
                        metadata_dict = json.loads(raw)
                    except Exception:
                        pass

        if not result_image_dest.exists():
            raise ValueError("No output image found inside imported ZIP archive")

        job.result_image_path = str(result_image_dest.resolve())
        job.result_image_url = f"/storage/results/{result_image_dest.name}"
        job.status = JobStatus.COMPLETED

        if metadata_dict:
            try:
                job.generation_metadata = GenerationMetadata.model_validate(metadata_dict)
            except Exception:
                pass

        cls.save_job(job)
        return job
