import io
import zipfile
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.models.schemas import JobStatus

client = TestClient(app)

def create_test_image_bytes():
    buf = io.BytesIO()
    img = Image.new("RGB", (300, 300), color=(255, 255, 255))
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_full_api_workflow():
    # 1. Upload sketch
    buf = create_test_image_bytes()
    upload_res = client.post(
        "/api/upload",
        files={"file": ("sketch.png", buf, "image/png")}
    )
    assert upload_res.status_code == 200
    job_data = upload_res.json()
    job_id = job_data["id"]
    assert job_data["status"] == JobStatus.CREATED

    # 2. Preprocess sketch
    preprocess_res = client.post(
        f"/api/preprocess/{job_id}",
        json={
            "low_threshold": 100,
            "high_threshold": 200,
            "blur_kernel": 3,
            "denoise_strength": 5,
            "target_size": 512
        }
    )
    assert preprocess_res.status_code == 200
    pre_data = preprocess_res.json()
    assert pre_data["width"] == 512
    assert "preprocessed_url" in pre_data

    # 3. Export Kaggle bundle
    export_res = client.get(f"/api/jobs/{job_id}/export-bundle")
    assert export_res.status_code == 200
    assert export_res.headers["content-type"] == "application/zip"
    
    # Verify bundle contents
    zip_bytes = io.BytesIO(export_res.content)
    with zipfile.ZipFile(zip_bytes, "r") as zf:
        namelist = zf.namelist()
        assert "control_sketch.png" in namelist
        assert "manifest.json" in namelist

    # 4. Run local inference (fallback mode)
    inf_res = client.post(
        f"/api/inference/{job_id}",
        json={
            "prompt_options": {
                "scene_description": "theatrical auditorium with stone pillars",
                "materials": ["polished_marble"],
                "lighting": "dramatic_stage",
                "render_style": "theatrical_set"
            },
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
            "controlnet_conditioning_scale": 1.0,
            "seed": 42,
            "force_fallback": True
        }
    )
    assert inf_res.status_code == 200
    inf_data = inf_res.json()
    assert inf_data["status"] == JobStatus.COMPLETED
    assert inf_data["result_image_url"] is not None
    assert inf_data["generation_metadata"]["steps"] == 25

    # 5. Evaluate results
    eval_res = client.post(f"/api/eval/{job_id}?tolerance=2")
    assert eval_res.status_code == 200
    eval_data = eval_res.json()
    assert "edge_preservation_f1" in eval_data
    assert "structural_similarity_ssim" in eval_data

    # 6. Retrieve job
    get_res = client.get(f"/api/jobs/{job_id}")
    assert get_res.status_code == 200
    final_job = get_res.json()
    assert final_job["id"] == job_id
    assert final_job["evaluation"] is not None

def test_import_result_bundle(tmp_path):
    # Upload and preprocess first
    buf = create_test_image_bytes()
    upload_res = client.post("/api/upload", files={"file": ("sketch2.png", buf, "image/png")})
    job_id = upload_res.json()["id"]

    client.post(
        f"/api/preprocess/{job_id}",
        json={"low_threshold": 80, "high_threshold": 160, "target_size": 512}
    )

    # Create mock Kaggle result bundle zip
    result_zip_path = tmp_path / "kaggle_output.zip"
    dummy_img = Image.new("RGB", (512, 512), color=(100, 150, 200))
    dummy_img_bytes = io.BytesIO()
    dummy_img.save(dummy_img_bytes, format="PNG")
    
    with zipfile.ZipFile(result_zip_path, "w") as zf:
        zf.writestr("job_result.png", dummy_img_bytes.getvalue())
        zf.writestr("metadata.json", '{"prompt": "kaggle test", "negative_prompt": "low quality", "steps": 30, "guidance_scale": 7.5, "controlnet_conditioning_scale": 1.0, "seed": 99, "inference_time_seconds": 4.2, "device": "cuda:0", "model_id": "sd15", "controlnet_id": "canny"}')

    with open(result_zip_path, "rb") as f:
        import_res = client.post(
            f"/api/jobs/{job_id}/import-result",
            files={"file": ("kaggle_output.zip", f, "application/zip")}
        )
    assert import_res.status_code == 200
    res_job = import_res.json()
    assert res_job["status"] == JobStatus.COMPLETED
    assert res_job["generation_metadata"]["device"] == "cuda:0"
