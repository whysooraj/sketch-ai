import cv2
import numpy as np

from app.services.evaluator import SceneEvaluator

def test_scene_evaluator_metrics(tmp_path):
    sketch_edge_path = tmp_path / "sketch_edge.png"
    result_path = tmp_path / "result.png"

    # Edge map
    edge_map = np.zeros((512, 512), dtype=np.uint8)
    cv2.rectangle(edge_map, (100, 100), (400, 400), 255, 2)
    cv2.imwrite(str(sketch_edge_path), edge_map)

    # Result image that preserves the rectangle with shaded texture
    result_img = np.full((512, 512, 3), 180, dtype=np.uint8)
    cv2.rectangle(result_img, (100, 100), (400, 400), (40, 40, 40), 2)
    cv2.imwrite(str(result_path), result_img)

    metrics = SceneEvaluator.evaluate_render(sketch_edge_path, result_path, tolerance_radius=2)
    assert 0.0 <= metrics.edge_preservation_f1 <= 1.0
    assert 0.0 <= metrics.edge_preservation_iou <= 1.0
    assert -1.0 <= metrics.structural_similarity_ssim <= 1.0
    assert metrics.evaluation_time_seconds >= 0.0
    assert metrics.details["tolerance_radius_pixels"] == 2
