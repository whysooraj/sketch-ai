import pytest
from pathlib import Path
import cv2
import numpy as np

from app.services.preprocessor import SketchPreprocessor
from app.models.schemas import PreprocessRequest

def test_sketch_preprocessor_letterbox():
    # Test letterboxing an oblong 400x200 image into 512x512
    img = np.ones((200, 400), dtype=np.uint8) * 255
    boxed, scale, (top, left) = SketchPreprocessor.letterbox_image(img, target_size=512)
    assert boxed.shape == (512, 512)
    assert scale == 512 / 400
    assert top > 0
    assert left == 0

def test_sketch_preprocessing_pipeline(tmp_path):
    # Create test sketch
    sketch_path = tmp_path / "test_sketch.png"
    canvas = np.full((300, 300, 3), 250, dtype=np.uint8)
    cv2.rectangle(canvas, (50, 50), (250, 250), (20, 20, 20), 3)
    cv2.imwrite(str(sketch_path), canvas)

    params = PreprocessRequest(
        low_threshold=50,
        high_threshold=150,
        blur_kernel=3,
        denoise_strength=5,
        target_size=512
    )

    result = SketchPreprocessor.process_sketch(sketch_path, "job_test_01", params)
    assert result.width == 512
    assert result.height == 512
    assert result.edge_pixel_count > 0
    assert result.edge_density > 0.0

    edge_file = Path(result.preprocessed_url.replace("/storage/preprocessed/", "backend/app/storage/preprocessed/"))
    assert edge_file.exists()

def test_sketch_preprocessor_invalid_dimensions(tmp_path):
    tiny_path = tmp_path / "tiny.png"
    tiny = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(tiny_path), tiny)

    params = PreprocessRequest()
    with pytest.raises(ValueError, match="too small"):
        SketchPreprocessor.process_sketch(tiny_path, "job_tiny", params)
