from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings, STORAGE_DIR
from app.routers import preprocess, inference, jobs, eval

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Backend API for Sketch-to-Photorealistic 2D Scene Renderer (SD 1.5 + ControlNet Canny)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount local file storage for previewing uploaded sketches, edge maps, and generated renders
app.mount("/storage", StaticFiles(directory=str(STORAGE_DIR)), name="storage")

# Include routers
app.include_router(preprocess.router)
app.include_router(inference.router)
app.include_router(jobs.router)
app.include_router(eval.router)

@app.get("/")
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.version,
        "storage_ready": STORAGE_DIR.exists()
    }
