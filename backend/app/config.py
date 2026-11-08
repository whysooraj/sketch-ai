from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
PREPROCESSED_DIR = STORAGE_DIR / "preprocessed"
RESULTS_DIR = STORAGE_DIR / "results"
JOBS_DIR = STORAGE_DIR / "jobs"
BUNDLES_DIR = STORAGE_DIR / "bundles"

for directory in [UPLOADS_DIR, PREPROCESSED_DIR, RESULTS_DIR, JOBS_DIR, BUNDLES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

class Settings(BaseModel):
    app_name: str = "Sketch-to-Photorealistic Scene Renderer"
    version: str = "1.0.0"
    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*"
    ]
    max_upload_size_bytes: int = 15 * 1024 * 1024
    allowed_mime_types: list[str] = ["image/jpeg", "image/png", "image/webp"]
    default_image_size: int = 512
    inference_mode: str = "cpu_fallback"

settings = Settings()
