from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class JobStatus(str, Enum):
    CREATED = "CREATED"
    PREPROCESSED = "PREPROCESSED"
    PENDING_GPU = "PENDING_GPU"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class PreprocessRequest(BaseModel):
    low_threshold: int = Field(default=100, ge=0, le=255, description="Canny lower hysteresis threshold")
    high_threshold: int = Field(default=200, ge=0, le=255, description="Canny upper hysteresis threshold")
    blur_kernel: int = Field(default=3, ge=1, le=15, description="Gaussian blur kernel size (odd)")
    denoise_strength: int = Field(default=5, ge=0, le=30, description="Denoising filter strength")
    invert_input: bool = Field(default=False, description="Explicitly invert input before processing")
    auto_detect_inversion: bool = Field(default=True, description="Auto detect dark sketch on light background")
    target_size: int = Field(default=512, ge=256, le=1024, description="Output edge map square dimension")

class PreprocessResult(BaseModel):
    job_id: str
    original_url: str
    preprocessed_url: str
    width: int
    height: int
    edge_pixel_count: int
    edge_density: float
    parameters: PreprocessRequest

class PromptBuilderOptions(BaseModel):
    scene_description: str = Field(..., min_length=3, description="Core scene setting, architecture, or stage design")
    materials: List[str] = Field(default_factory=list, description="Specific materials (e.g., aged oak wood, polished concrete)")
    lighting: str = Field(default="dramatic stage lighting", description="Lighting conditions and atmosphere")
    render_style: str = Field(default="architectural photograph", description="Desired rendering aesthetic")
    negative_prompt: Optional[str] = Field(
        default="blurry, low quality, cartoon, drawing, painting, sketch, distorted geometry, lowres, watermark",
        description="Concepts to avoid in render"
    )
    custom_modifiers: Optional[str] = Field(default=None, description="Optional extra prompt keywords")

class InferenceRequest(BaseModel):
    prompt_options: PromptBuilderOptions
    num_inference_steps: int = Field(default=30, ge=10, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    controlnet_conditioning_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    seed: Optional[int] = Field(default=42)
    force_fallback: bool = Field(default=False, description="Use CPU fallback engine for rapid local verification")

class GenerationMetadata(BaseModel):
    prompt: str
    negative_prompt: str
    steps: int
    guidance_scale: float
    controlnet_conditioning_scale: float
    seed: int
    inference_time_seconds: float
    device: str
    model_id: str
    controlnet_id: str

class EvaluationMetrics(BaseModel):
    edge_preservation_f1: float = Field(description="Dice/F1 score between sketch edges and render edges")
    edge_preservation_iou: float = Field(description="IoU between sketch edges and render edges")
    structural_similarity_ssim: float = Field(description="SSIM score against normalized structure")
    evaluation_time_seconds: float
    details: Dict[str, Any] = Field(default_factory=dict)

class Job(BaseModel):
    id: str
    status: JobStatus
    created_at: str
    updated_at: str
    original_image_path: Optional[str] = None
    original_image_url: Optional[str] = None
    preprocessed_image_path: Optional[str] = None
    preprocessed_image_url: Optional[str] = None
    result_image_path: Optional[str] = None
    result_image_url: Optional[str] = None
    preprocess_config: Optional[Dict[str, Any]] = None
    prompt_config: Optional[Dict[str, Any]] = None
    generation_metadata: Optional[GenerationMetadata] = None
    evaluation: Optional[EvaluationMetrics] = None
    error_message: Optional[str] = None

class JobResponse(BaseModel):
    job: Job

class JobListResponse(BaseModel):
    jobs: List[Job]
