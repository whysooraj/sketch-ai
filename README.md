# Sketch-to-Photorealistic 2D Scene Renderer

Academic system that transforms hand-drawn theatrical set-design sketches into photorealistic 2D scene renders while strictly preserving spatial structures, architectural contours, and perspective geometry.

## Architecture & Technology Stack

The pipeline integrates edge detection with conditioned latent diffusion:

```
[Hand-Drawn Sketch]
         |
         v
[Preprocessing & Validation] (Grayscale, Bilateral Denoising, Aspect-Preserving Letterbox)
         |
         v
[Canny Edge Map Extraction] (Normalized 512x512 binary control contours)
         |
         v
[Prompt Synthesis Engine] (Set architecture, tactile materials, volumetric lighting, photorealistic styles)
         |
         +-------------------------------------+
         |                                     |
         v                                     v
[Local CPU Fallback / Mock Engine]    [Standalone Kaggle GPU Notebook]
(Instant local testing & dev)         (NVIDIA T4 / P100 Free Acceleration)
         |                                     |
         +-------------------------------------+
         |
         v
[Photorealistic 2D Render Output]
         |
         v
[Quantitative Evaluation] (Edge Preservation F1, IoU, Structural SSIM, Generation Latency)
```

### Core Technologies
- Backend: Python 3.11, FastAPI, OpenCV, Pillow, Pytest, scikit-image
- AI Models: Stable Diffusion 1.5 (`runwayml/stable-diffusion-v1-5`), ControlNet Canny (`lllyasviel/sd-controlnet-canny`)
- Frontend: Next.js 14+ (App Router), TypeScript, Tailwind CSS, Lucide Icons
- Compute Strategy: Zero-cost architecture (₹0 spend). Local CPU execution for preprocessing, evaluation, UI, and fallback rendering; free Kaggle GPU sessions for heavy diffusion inference.

## Hard Constraints Compliance

1. Total budget: ₹0.
2. No local dedicated GPU required; runs smoothly on CPU.
3. No paid cloud services, billing accounts, or paid API keys used.
4. Uses free open-source pretrained weights from Hugging Face Hub.
5. No models trained from scratch.
6. Standalone Kaggle GPU notebook provided for free T4/P100 inference sessions.
7. Local full-stack application operates independently with high-fidelity CPU fallback and seamless Kaggle bundle import/export.

## Directory Structure

```
sketch-ai/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI entry point & CORS configuration
│   │   ├── config.py                # Environment and storage configuration
│   │   ├── models/schemas.py        # Pydantic data schemas
│   │   ├── routers/
│   │   │   ├── preprocess.py        # Upload and edge extraction endpoints
│   │   │   ├── inference.py         # Diffusion inference router
│   │   │   ├── jobs.py              # Job state manager & bundle transfer
│   │   │   └── eval.py              # Quantitative evaluation router
│   │   └── services/
│   │       ├── preprocessor.py      # OpenCV Canny pipeline & letterboxing
│   │       ├── prompt_builder.py    # Structured scene & material prompt assembler
│   │       ├── inference_engine.py  # ControlNet SD 1.5 + CPU fallback engine
│   │       ├── job_manager.py       # Persistence & zip bundle packaging
│   │       ├── evaluator.py         # Edge preservation (F1/IoU) & SSIM metrics
│   │       └── benchmark_suite.py   # Automated evaluation benchmark
│   ├── storage/                     # Local persistence store (uploads, edges, results)
│   ├── tests/                       # Complete pytest unit and integration suite
│   └── requirements.txt
├── frontend/                        # Next.js 14+ React TypeScript application
│   ├── src/
│   │   ├── app/page.tsx             # Interactive workspace & tabbed layout
│   │   ├── components/              # Comparison slider, upload, controls, metrics
│   │   └── types/index.ts           # Frontend TypeScript interfaces
│   └── package.json
├── kaggle/
│   ├── sketch_to_photorealistic_inference.ipynb # Standalone Kaggle GPU notebook
│   └── README.md
├── samples/
│   ├── sample_stage_sketch.png      # Academic reference architectural sketch
│   └── generate_sample.py
├── docs/
│   ├── EVALUATION_REPORT.md         # Quantitative evaluation benchmark output
│   └── KAGGLE_GUIDE.md             # Free GPU execution guide
└── AGENTS.md                        # Project requirements and hard constraints
```

## Setup and Local Execution

### 1. Prerequisites
- Linux / macOS / Windows with WSL
- Python 3.11+
- Node.js 18+ and npm

### 2. Backend Setup
```bash
# Create virtual environment and install dependencies
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r backend/requirements.txt

# Run the test suite
PYTHONPATH=backend pytest backend/tests -v

# Start the FastAPI server
PYTHONPATH=backend uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be accessible at `http://localhost:8000` with Swagger documentation at `http://localhost:8000/docs`.

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run build
npm run start # or npm run dev for development
```

The web application will be accessible at `http://localhost:3000`.

## Kaggle GPU Inference Workflow

1. In the web interface, upload a hand-drawn sketch and run the preprocessor to generate Canny edges.
2. Configure materials, lighting, and camera style.
3. Click **Export Bundle** to download `job_<id>_kaggle_bundle.zip`.
4. Open Kaggle and import `kaggle/sketch_to_photorealistic_inference.ipynb`.
5. Set Accelerator to `GPU T4 x2` and Internet to `ON`.
6. Upload the bundle to Kaggle, run the inference cells, and download `kaggle_results_bundle.zip`.
7. Click **Import Result** in the local web application. The application will render the scene, activate the interactive before/after split slider, and display quantitative evaluation metrics.

## Quantitative Evaluation Benchmark

Evaluated on the reference theater stage set design:

| Scenario | Edge Preservation (F1) | IoU | Structural SSIM | Latency |
|---|---|---|---|---|
| Classical Theater (Oak + Marble, Dramatic Stage) | 79.48% | 65.95% | 0.8164 | 0.118s |
| Ancient Temple Ruins (Stone + Daylight) | 75.07% | 60.09% | 0.8166 | 0.128s |
| Moody Opera Stage (Velvet + Chiaroscuro) | 82.26% | 69.87% | 0.8135 | 0.119s |

Metric details and methodology are documented in `docs/EVALUATION_REPORT.md`.

## Known Limitations

1. High sketch complexity (dense micro-hatching) requires tuning the Canny upper threshold to avoid noisy edge maps.
2. Extreme non-square aspect ratios receive letterboxing padding with black boundary pixels to avoid distorting spatial geometry.
3. Kaggle free GPU sessions require manual upload and download of job zip bundles when remote webhooks or public tunnels are not configured.
