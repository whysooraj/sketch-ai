# Kaggle Free GPU Execution Guide

## Overview

This guide provides step-by-step instructions for executing the heavy Stable Diffusion 1.5 + ControlNet Canny inference on Kaggle's free GPU instances (NVIDIA T4 x2 or P100), fulfilling the ₹0 budget constraint.

## Step 1: Kaggle Account & Setup

1. Create a free account at https://www.kaggle.com.
2. Complete phone verification under your Kaggle account settings. This unlocks GPU accelerators and outbound internet access.
3. Kaggle provides 30 hours of free GPU compute per week.

## Step 2: Upload the Notebook

1. Navigate to https://www.kaggle.com/code and click **New Notebook**.
2. Select **File -> Import Notebook**.
3. Select and upload `kaggle/sketch_to_photorealistic_inference.ipynb` from this repository.

## Step 3: Configure Notebook Runtime

In the right-hand **Notebook options** panel:
- **Accelerator**: `GPU T4 x2` or `GPU P100`.
- **Internet**: Set to **ON** (required to fetch pretrained weights from Hugging Face).
- **Environment**: Default (latest).

## Step 4: Sketch Data Transfer Options

### Workflow A: Local Job Bundle (Recommended)

1. In the local web application (http://localhost:3000), upload your hand-drawn sketch.
2. Run preprocessing to generate the Canny edge map and configure prompts/materials.
3. Click **Export Bundle**. A zip archive named `job_<id>_kaggle_bundle.zip` will download.
4. In Kaggle, upload this zip file to `/kaggle/working/job_bundle.zip`.
5. In Section 5 of the notebook, set `USE_BUNDLE = True`.
6. Run the notebook. The pipeline automatically extracts the edge map, applies user prompt configurations, and executes inference.

### Workflow B: Direct Upload

1. Upload any raw sketch image into `/kaggle/working/sketch.png`.
2. Keep `USE_BUNDLE = False` and customize the prompt parameters directly in the notebook cell.

## Step 5: Download Results and Import to Local Application

1. After running the generation and evaluation cells in the notebook, `kaggle_results_bundle.zip` will be produced in `/kaggle/working/`.
2. In the Kaggle file explorer pane, click the three dots next to `kaggle_results_bundle.zip` and select **Download**.
3. Return to the local web application on the active job screen, click **Import Result**, and select the downloaded zip file.
4. The local application updates the job status to `COMPLETED`, displays the generated photorealistic scene, activates the interactive before/after split slider, and displays structural metrics.
