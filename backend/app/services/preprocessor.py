from pathlib import Path
from typing import Tuple, Dict, Any
import cv2
import numpy as np
from PIL import Image

from app.config import PREPROCESSED_DIR, UPLOADS_DIR
from app.models.schemas import PreprocessRequest, PreprocessResult

class SketchPreprocessor:
    @staticmethod
    def validate_image_file(image_path: Path, max_size_bytes: int = 15 * 1024 * 1024) -> None:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")
        size = image_path.stat().st_size
        if size > max_size_bytes:
            raise ValueError(f"File size {size} exceeds limit of {max_size_bytes} bytes")
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image format: {e}")

    @classmethod
    def letterbox_image(cls, image: np.ndarray, target_size: int = 512, pad_value: int = 0) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        h, w = image.shape[:2]
        scale = min(target_size / w, target_size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
        
        if len(image.shape) == 3:
            canvas = np.full((target_size, target_size, image.shape[2]), pad_value, dtype=np.uint8)
        else:
            canvas = np.full((target_size, target_size), pad_value, dtype=np.uint8)
            
        top = (target_size - nh) // 2
        left = (target_size - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        
        return canvas, scale, (top, left)

    @classmethod
    def process_sketch(
        cls,
        image_path: Path,
        job_id: str,
        params: PreprocessRequest
    ) -> PreprocessResult:
        cls.validate_image_file(image_path)

        # 1. Read image with OpenCV
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError("Failed to decode image with OpenCV")

        orig_h, orig_w = bgr.shape[:2]
        if orig_h < 64 or orig_w < 64:
            raise ValueError(f"Image dimensions ({orig_w}x{orig_h}) too small; minimum is 64x64")
        if orig_h > 4096 or orig_w > 4096:
            raise ValueError(f"Image dimensions ({orig_w}x{orig_h}) too large; maximum is 4096x4096")

        # 2. Convert to Grayscale
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 3. Handle Inversion / Light background
        mean_intensity = float(np.mean(gray))
        is_light_background = mean_intensity > 127.0

        working_gray = gray.copy()
        if params.invert_input:
            working_gray = cv2.bitwise_not(working_gray)

        # 4. Denoise and smoothing to remove paper grain and compression artifacts
        kernel_size = params.blur_kernel if params.blur_kernel % 2 == 1 else params.blur_kernel + 1
        if params.denoise_strength > 0:
            # Bilateral filter preserves structural lines while smoothing texture
            d = max(3, min(9, params.denoise_strength * 2))
            denoised = cv2.bilateralFilter(working_gray, d=d, sigmaColor=75, sigmaSpace=75)
        else:
            denoised = working_gray

        smoothed = cv2.GaussianBlur(denoised, (kernel_size, kernel_size), 0)

        # 5. Extract Canny edges
        # ControlNet Canny expects white lines (255) on black background (0)
        edges = cv2.Canny(smoothed, params.low_threshold, params.high_threshold)

        # 6. Normalize and letterbox resize to target_size (512x512)
        letterboxed_edges, scale, (pad_top, pad_left) = cls.letterbox_image(edges, target_size=params.target_size, pad_value=0)

        # Calculate edge density metrics
        edge_pixel_count = int(np.count_nonzero(letterboxed_edges))
        total_pixels = params.target_size * params.target_size
        edge_density = float(edge_pixel_count / total_pixels)

        # Save preprocessed edge map
        output_filename = f"{job_id}_edge.png"
        output_path = PREPROCESSED_DIR / output_filename
        cv2.imwrite(str(output_path), letterboxed_edges)

        return PreprocessResult(
            job_id=job_id,
            original_url=f"/storage/uploads/{image_path.name}",
            preprocessed_url=f"/storage/preprocessed/{output_filename}",
            width=params.target_size,
            height=params.target_size,
            edge_pixel_count=edge_pixel_count,
            edge_density=round(edge_density, 5),
            parameters=params
        )
