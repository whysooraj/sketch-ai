import time
from pathlib import Path
from typing import Dict, Any
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from app.models.schemas import EvaluationMetrics

class SceneEvaluator:
    @classmethod
    def evaluate_render(
        cls,
        sketch_edge_path: Path,
        result_image_path: Path,
        tolerance_radius: int = 2
    ) -> EvaluationMetrics:
        start_time = time.time()

        if not sketch_edge_path.exists():
            raise FileNotFoundError(f"Sketch edge file not found at {sketch_edge_path}")
        if not result_image_path.exists():
            raise FileNotFoundError(f"Result render file not found at {result_image_path}")

        # 1. Load preprocessed sketch edge map
        sketch_edges = cv2.imread(str(sketch_edge_path), cv2.IMREAD_GRAYSCALE)
        if sketch_edges is None:
            raise ValueError("Failed to load sketch edge image")

        # 2. Load generated scene render and convert to grayscale
        render_bgr = cv2.imread(str(result_image_path))
        if render_bgr is None:
            raise ValueError("Failed to load rendered image")
        render_gray = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2GRAY)

        # Match dimensions if needed
        h, w = sketch_edges.shape
        if render_gray.shape != (h, w):
            render_gray = cv2.resize(render_gray, (w, h), interpolation=cv2.INTER_AREA)

        # 3. Extract edges from rendered image with adaptive thresholds
        med = float(np.median(render_gray))
        low_t = int(max(15, (1.0 - 0.33) * med))
        high_t = int(min(240, (1.0 + 0.33) * med))
        if low_t >= high_t:
            low_t, high_t = 30, 100
        render_edges = cv2.Canny(render_gray, low_t, high_t)

        # 4. Compute Edge Preservation (IoU and Dice/F1 with dilation tolerance)
        sketch_bin = (sketch_edges > 127).astype(np.uint8)
        render_bin = (render_edges > 127).astype(np.uint8)

        # Tolerance kernel for natural artistic stroke thickness variations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance_radius + 1, 2 * tolerance_radius + 1))
        sketch_dilated = cv2.dilate(sketch_bin, kernel, iterations=1)
        render_dilated = cv2.dilate(render_bin, kernel, iterations=1)

        # True Positives: sketch edges matched within tolerance of render edges
        matched_sketch = np.logical_and(sketch_bin, render_dilated)
        matched_render = np.logical_and(render_bin, sketch_dilated)

        tp_sketch = np.sum(matched_sketch)
        total_sketch_edges = np.sum(sketch_bin)
        total_render_edges = np.sum(render_bin)

        if total_sketch_edges + total_render_edges == 0:
            precision = 1.0
            recall = 1.0
            f1_score = 1.0
            iou = 1.0
        else:
            precision = float(tp_sketch / max(1, total_render_edges))
            recall = float(tp_sketch / max(1, total_sketch_edges))
            f1_score = float(2 * precision * recall / max(1e-6, precision + recall))
            iou = float(tp_sketch / max(1, total_sketch_edges + total_render_edges - tp_sketch))

        # 5. Compute Structural Similarity (SSIM)
        # Compare high-frequency edge structure
        score_ssim, _ = ssim(sketch_edges, render_edges, full=True)

        eval_time = round(time.time() - start_time, 3)

        return EvaluationMetrics(
            edge_preservation_f1=round(float(np.clip(f1_score, 0.0, 1.0)), 4),
            edge_preservation_iou=round(float(np.clip(iou, 0.0, 1.0)), 4),
            structural_similarity_ssim=round(float(np.clip(score_ssim, -1.0, 1.0)), 4),
            evaluation_time_seconds=eval_time,
            details={
                "sketch_edge_pixels": int(total_sketch_edges),
                "render_edge_pixels": int(total_render_edges),
                "tolerance_radius_pixels": tolerance_radius,
                "precision": round(precision, 4),
                "recall": round(recall, 4)
            }
        )
