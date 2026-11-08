from pathlib import Path
import numpy as np
import cv2

from app.services.inference_engine import InferenceEngine
from app.models.schemas import InferenceRequest, PromptBuilderOptions

def test_fallback_inference_engine(tmp_path):
    # Create synthetic edge map
    edge_path = tmp_path / "edge_map.png"
    edge_map = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(edge_map, (256, 256), 100, 255, 3)
    cv2.imwrite(str(edge_path), edge_map)

    req = InferenceRequest(
        prompt_options=PromptBuilderOptions(
            scene_description="spherical sculpture in marble hall",
            materials=["polished_marble"],
            lighting="golden_hour",
            render_style="architectural_photo"
        ),
        num_inference_steps=20,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.0,
        seed=123,
        force_fallback=True
    )

    out_path, meta = InferenceEngine.run_inference(edge_path, "job_test_inf", req)
    assert out_path.exists()
    assert meta.seed == 123
    assert meta.steps == 20
    assert meta.inference_time_seconds > 0.0

    # Verify output image format
    res_img = cv2.imread(str(out_path))
    assert res_img.shape == (512, 512, 3)
