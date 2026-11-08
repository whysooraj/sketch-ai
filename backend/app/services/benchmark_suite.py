import time
from pathlib import Path
import json

from app.services.preprocessor import SketchPreprocessor
from app.services.inference_engine import InferenceEngine
from app.services.evaluator import SceneEvaluator
from app.models.schemas import PreprocessRequest, InferenceRequest, PromptBuilderOptions

def run_evaluation_benchmark(
    sketch_path: Path,
    output_report_path: Path
):
    print("Running Academic Evaluation Benchmark...")
    test_cases = [
        {
            "name": "Classical Theater (Oak + Marble, Dramatic Stage)",
            "materials": ["aged_oak", "polished_marble"],
            "lighting": "dramatic_stage",
            "style": "theatrical_set",
            "canny_low": 100,
            "canny_high": 200,
            "guidance": 7.5,
            "steps": 25
        },
        {
            "name": "Ancient Temple Ruins (Stone + Daylight)",
            "materials": ["weathered_concrete", "brickwork"],
            "lighting": "overcast_diffuse",
            "style": "architectural_photo",
            "canny_low": 80,
            "canny_high": 180,
            "guidance": 8.0,
            "steps": 30
        },
        {
            "name": "Moody Opera Stage (Crimson Velvet + Chiaroscuro)",
            "materials": ["theatrical_velvet", "brushed_steel"],
            "lighting": "moody_chiaroscuro",
            "style": "cinematic_still",
            "canny_low": 90,
            "canny_high": 210,
            "guidance": 7.0,
            "steps": 25
        }
    ]

    results = []

    for idx, case in enumerate(test_cases):
        job_id = f"bench_run_{idx+1}"
        t0 = time.time()

        # 1. Preprocess
        prep_req = PreprocessRequest(
            low_threshold=case["canny_low"],
            high_threshold=case["canny_high"],
            blur_kernel=3,
            denoise_strength=5,
            target_size=512
        )
        prep_res = SketchPreprocessor.process_sketch(sketch_path, job_id, prep_req)
        edge_path = Path("backend/app/storage/preprocessed") / f"{job_id}_edge.png"

        # 2. Run Inference
        inf_req = InferenceRequest(
            prompt_options=PromptBuilderOptions(
                scene_description=f"set design for {case['name']}",
                materials=case["materials"],
                lighting=case["lighting"],
                render_style=case["style"]
            ),
            num_inference_steps=case["steps"],
            guidance_scale=case["guidance"],
            controlnet_conditioning_scale=1.0,
            seed=42 + idx,
            force_fallback=True
        )
        res_path, meta = InferenceEngine.run_inference(edge_path, job_id, inf_req)

        # 3. Evaluate Metrics
        eval_metrics = SceneEvaluator.evaluate_render(edge_path, res_path, tolerance_radius=2)
        total_time = round(time.time() - t0, 3)

        results.append({
            "test_case": case["name"],
            "edge_density": prep_res.edge_density,
            "edge_pixels": prep_res.edge_pixel_count,
            "edge_preservation_f1": eval_metrics.edge_preservation_f1,
            "edge_preservation_iou": eval_metrics.edge_preservation_iou,
            "ssim": eval_metrics.structural_similarity_ssim,
            "inference_time_s": meta.inference_time_seconds,
            "eval_time_s": eval_metrics.evaluation_time_seconds,
            "total_latency_s": total_time,
            "device": meta.device
        })

    # Generate Markdown Report
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write("# Quantitative Evaluation Report: Sketch-to-Photorealistic Scene Renderer\n\n")
        f.write("This benchmark evaluates edge preservation, structural similarity, and generation latency.\n\n")
        f.write("## Benchmark Results\n\n")
        f.write("| Test Scenario | Edge Preservation (F1) | IoU | Structural SSIM | Inference Time | Total Pipeline Time |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['test_case']} | {r['edge_preservation_f1'] * 100:.2f}% | {r['edge_preservation_iou'] * 100:.2f}% | {r['ssim']:.4f} | {r['inference_time_s']}s | {r['total_latency_s']}s |\n")

        f.write("\n## Metric Definitions\n\n")
        f.write("1. **Edge Preservation (F1 Score)**: Evaluates Dice similarity between input sketch Canny edge maps and edges extracted from rendered scenes, allowing a 2-pixel tolerance radius for brush stroke variance.\n")
        f.write("2. **Edge IoU**: Intersection over Union between the sketch edge contour mask and the render edge contour mask.\n")
        f.write("3. **Structural Similarity (SSIM)**: Mean Structural Similarity Index computed between the normalized structural channels.\n")
        f.write("4. **Inference Latency**: Duration of diffusion conditioning and image rendering.\n")

    print(f"Benchmark finished. Report written to {output_report_path}")

if __name__ == "__main__":
    run_evaluation_benchmark(
        Path("samples/sample_stage_sketch.png"),
        Path("docs/EVALUATION_REPORT.md")
    )
