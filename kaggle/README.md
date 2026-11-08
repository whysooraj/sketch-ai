# Kaggle GPU Inference Guide

This document explains how to run the photorealistic inference pipeline on free Kaggle GPUs without paid cloud services or local GPU hardware.

## Prerequisites

- Free Kaggle Account (https://www.kaggle.com)
- Phone verification on Kaggle (required once to enable internet access and GPU acceleration)
- Total monetary cost: ₹0

## Step-by-Step Instructions

### 1. Create a New Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **New Notebook**.
2. Click **File -> Import Notebook** and upload `kaggle/sketch_to_photorealistic_inference.ipynb` from this repository.

### 2. Configure Notebook Session Settings

In the right-hand panel (**Notebook options**):
- **Accelerator**: Select `GPU T4 x2` or `GPU P100`.
- **Internet**: Toggle to `ON`.
- **Language**: `Python`.
- **Persistence**: `No persistence` (or `Files only`).

### 3. Transferring Sketch from Local Web Application

Option A — **Job Bundle Workflow (Recommended)**:
1. In the local web application, upload and preprocess your sketch.
2. Click **Export for Kaggle GPU**. This downloads `job_<id>_kaggle_bundle.zip`.
3. In Kaggle, click the upload icon in the notebook files pane (or run cell with uploaded file) and place `job_bundle.zip` in `/kaggle/working/`.
4. Set `USE_BUNDLE = True` in Section 5.

Option B — **Direct Sketch Upload**:
1. Upload any raw sketch image (PNG/JPEG) directly into `/kaggle/working/sketch.png`.
2. Configure prompt, materials, lighting, and style in Section 5.

### 4. Execute the Inference Cells

1. Run **Section 1**: Installs free dependencies (`diffusers`, `transformers`, `accelerate`, etc.).
2. Run **Section 2 & 3**: Verifies CUDA and loads pretrained weights (`runwayml/stable-diffusion-v1-5` + `lllyasviel/sd-controlnet-canny`) with FP16 and CPU offloading.
3. Run **Section 5 & 6**: Executes ControlNet Canny diffusion inference.
4. Run **Section 7**: Computes quantitative edge preservation (F1 and SSIM) and displays side-by-side comparison.
5. Run **Section 8**: Generates `kaggle_results_bundle.zip`.

### 5. Transferring Result Back to Local Application

1. In the Kaggle file explorer (right pane under `/kaggle/working/`), find `kaggle_results_bundle.zip` and click the three dots -> **Download**.
2. In the local web application on the active job page, click **Import Kaggle Result** and select `kaggle_results_bundle.zip`.
3. The local web application will instantly load the photorealistic render, update job status to `COMPLETED`, display the interactive before/after split slider, and record generation metadata.
