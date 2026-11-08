import json
from pathlib import Path

def generate_kaggle_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Sketch-to-Photorealistic 2D Scene Renderer\n",
                    "### Free GPU Inference Pipeline (Stable Diffusion 1.5 + ControlNet Canny)\n",
                    "\n",
                    "This standalone Kaggle notebook executes heavy diffusion inference on free Kaggle GPUs (NVIDIA T4 or P100).\n",
                    "\n",
                    "**Prerequisites on Kaggle:**\n",
                    "1. Kaggle Notebook settings -> **Accelerator**: Select `GPU T4 x2` or `GPU P100`.\n",
                    "2. Kaggle Notebook settings -> **Internet**: Turn **ON** (required to download HuggingFace pretrained weights).\n",
                    "3. Total Compute Cost: **₹0**."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Install Dependencies"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Install free open-source dependencies\n",
                    "!pip install -q diffusers==0.27.2 transformers==4.39.3 accelerate==0.28.0 controlnet-aux==0.0.7 opencv-python-headless scikit-image gradio"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Check GPU Hardware"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import torch\n",
                    "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
                    "if torch.cuda.is_available():\n",
                    "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n",
                    "    print(f\"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\")\n",
                    "else:\n",
                    "    print(\"WARNING: Running on CPU. Switch accelerator to GPU T4 in Kaggle settings.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Load Pretrained SD 1.5 + ControlNet Canny Weights"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler\n",
                    "from PIL import Image\n",
                    "\n",
                    "MODEL_ID = \"runwayml/stable-diffusion-v1-5\"\n",
                    "CONTROLNET_ID = \"lllyasviel/sd-controlnet-canny\"\n",
                    "\n",
                    "print(\"Loading ControlNet Canny model...\")\n",
                    "controlnet = ControlNetModel.from_pretrained(\n",
                    "    CONTROLNET_ID,\n",
                    "    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32\n",
                    ")\n",
                    "\n",
                    "print(\"Loading Stable Diffusion 1.5 pipeline...\")\n",
                    "pipe = StableDiffusionControlNetPipeline.from_pretrained(\n",
                    "    MODEL_ID,\n",
                    "    controlnet=controlnet,\n",
                    "    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,\n",
                    "    safety_checker=None\n",
                    ")\n",
                    "\n",
                    "# High-speed multi-step scheduler\n",
                    "pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)\n",
                    "\n",
                    "if torch.cuda.is_available():\n",
                    "    pipe.enable_model_cpu_offload()\n",
                    "    pipe.enable_vae_slicing()\n",
                    "print(\"Pipeline initialized successfully!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Sketch Preprocessor Utility"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import cv2\n",
                    "import numpy as np\n",
                    "\n",
                    "def prepare_canny_edge_map(image_path_or_array, low_threshold=100, high_threshold=200, target_size=512):\n",
                    "    if isinstance(image_path_or_array, (str, os.PathLike)):\n",
                    "        img = cv2.imread(str(image_path_or_array))\n",
                    "    else:\n",
                    "        img = np.array(image_path_or_array)\n",
                    "        if len(img.shape) == 3 and img.shape[2] == 3:\n",
                    "            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)\n",
                    "\n",
                    "    # Convert to grayscale\n",
                    "    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n",
                    "    smoothed = cv2.GaussianBlur(gray, (3, 3), 0)\n",
                    "    edges = cv2.Canny(smoothed, low_threshold, high_threshold)\n",
                    "\n",
                    "    # Aspect preserving letterbox to target_size\n",
                    "    h, w = edges.shape[:2]\n",
                    "    scale = min(target_size / w, target_size / h)\n",
                    "    nw, nh = int(round(w * scale)), int(round(h * scale))\n",
                    "    resized = cv2.resize(edges, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)\n",
                    "\n",
                    "    canvas = np.zeros((target_size, target_size), dtype=np.uint8)\n",
                    "    top = (target_size - nh) // 2\n",
                    "    left = (target_size - nw) // 2\n",
                    "    canvas[top:top+nh, left:left+nw] = resized\n",
                    "\n",
                    "    return Image.fromarray(canvas)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Inference Execution & Parameter Configuration\n",
                    "\n",
                    "You can either:\n",
                    "- **Option A**: Upload `job_xxx_kaggle_bundle.zip` exported from the local web application and set `USE_BUNDLE = True`.\n",
                    "- **Option B**: Upload any sketch image file directly and customize the prompts below."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import json\n",
                    "import time\n",
                    "import zipfile\n",
                    "from pathlib import Path\n",
                    "\n",
                    "# Configuration\n",
                    "USE_BUNDLE = False   # Set True if uploading job bundle from local app\n",
                    "BUNDLE_PATH = \"job_bundle.zip\"\n",
                    "\n",
                    "# Fallback / Direct sketch settings\n",
                    "SKETCH_PATH = \"sketch.png\"  # Path to uploaded sketch\n",
                    "SCENE_DESCRIPTION = \"a classical grand theater stage with Greek columns, red curtains, and polished stage floor\"\n",
                    "MATERIALS = [\"aged dark oak wood floor\", \"polished white Carrara marble columns\", \"heavy crimson velvet curtains\"]\n",
                    "LIGHTING = \"dramatic theatrical spotlights, strong volumetric chiaroscuro and atmospheric stage haze\"\n",
                    "STYLE = \"hyperrealistic architectural photograph, 35mm lens, f/8, 8k resolution, photorealistic master shot\"\n",
                    "NEGATIVE_PROMPT = \"blurry, low quality, drawing, painting, cartoon, deformed geometry, ugly, watermark, text\"\n",
                    "\n",
                    "NUM_STEPS = 30\n",
                    "GUIDANCE_SCALE = 7.5\n",
                    "CONTROLNET_SCALE = 1.0\n",
                    "SEED = 42\n",
                    "\n",
                    "# Handle bundle extraction if present\n",
                    "if USE_BUNDLE and os.path.exists(BUNDLE_PATH):\n",
                    "    with zipfile.ZipFile(BUNDLE_PATH, 'r') as zf:\n",
                    "        zf.extractall(\"extracted_bundle\")\n",
                    "    SKETCH_PATH = \"extracted_bundle/control_sketch.png\"\n",
                    "    if os.path.exists(\"extracted_bundle/manifest.json\"):\n",
                    "        with open(\"extracted_bundle/manifest.json\") as mf:\n",
                    "            manifest = json.load(mf)\n",
                    "            p_conf = manifest.get(\"prompt_config\", {})\n",
                    "            SCENE_DESCRIPTION = p_conf.get(\"scene_description\", SCENE_DESCRIPTION)\n",
                    "            MATERIALS = p_conf.get(\"materials\", MATERIALS)\n",
                    "            LIGHTING = p_conf.get(\"lighting\", LIGHTING)\n",
                    "            STYLE = p_conf.get(\"render_style\", STYLE)\n",
                    "            print(f\"Loaded bundle config for job {manifest.get('job_id')}\")\n",
                    "\n",
                    "# Compose prompt\n",
                    "materials_str = ', '.join(MATERIALS) if isinstance(MATERIALS, list) else str(MATERIALS)\n",
                    "full_prompt = f\"{STYLE}, photorealistic rendering of {SCENE_DESCRIPTION}, authentic materials: {materials_str}, illuminated by {LIGHTING}, volumetric depth, 8k\"\n",
                    "print(f\"\\nFull prompt:\\n{full_prompt}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Run Diffusion Generation"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Verify or prepare input control edge image\n",
                    "if not os.path.exists(SKETCH_PATH):\n",
                    "    # Generate synthetic sample sketch if none uploaded yet\n",
                    "    print(\"No sketch uploaded yet. Generating demo architectural sketch...\")\n",
                    "    sample = np.full((512, 512, 3), 245, dtype=np.uint8)\n",
                    "    cv2.rectangle(sample, (60, 100), (120, 440), (30, 30, 30), 3)\n",
                    "    cv2.rectangle(sample, (390, 100), (450, 440), (30, 30, 30), 3)\n",
                    "    cv2.ellipse(sample, (256, 260), (140, 120), 0, 180, 360, (30, 30, 30), 3)\n",
                    "    cv2.line(sample, (40, 440), (472, 440), (30, 30, 30), 3)\n",
                    "    control_image = prepare_canny_edge_map(sample)\n",
                    "else:\n",
                    "    test_img = cv2.imread(SKETCH_PATH, cv2.IMREAD_GRAYSCALE)\n",
                    "    # If already a binary edge map (white on black)\n",
                    "    if np.mean(test_img) < 60 and np.max(test_img) > 200:\n",
                    "        control_image = Image.open(SKETCH_PATH).convert(\"RGB\").resize((512, 512))\n",
                    "    else:\n",
                    "        control_image = prepare_canny_edge_map(SKETCH_PATH)\n",
                    "\n",
                    "print(\"Executing ControlNet Canny Diffusion...\")\n",
                    "start_time = time.time()\n",
                    "generator = torch.manual_seed(SEED)\n",
                    "\n",
                    "render_result = pipe(\n",
                    "    prompt=full_prompt,\n",
                    "    negative_prompt=NEGATIVE_PROMPT,\n",
                    "    image=control_image,\n",
                    "    num_inference_steps=NUM_STEPS,\n",
                    "    guidance_scale=GUIDANCE_SCALE,\n",
                    "    controlnet_conditioning_scale=CONTROLNET_SCALE,\n",
                    "    generator=generator\n",
                    ").images[0]\n",
                    "\n",
                    "duration = round(time.time() - start_time, 2)\n",
                    "print(f\"Inference completed in {duration} seconds!\")\n",
                    "\n",
                    "# Save result image\n",
                    "os.makedirs(\"output\", exist_ok=True)\n",
                    "render_result.save(\"output/job_result.png\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Visual Inspection & Evaluation"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import matplotlib.pyplot as plt\n",
                    "from skimage.metrics import structural_similarity as ssim\n",
                    "\n",
                    "# Compute edge preservation score\n",
                    "ctrl_gray = np.array(control_image.convert('L'))\n",
                    "res_gray = cv2.cvtColor(np.array(render_result), cv2.COLOR_RGB2GRAY)\n",
                    "res_edges = cv2.Canny(res_gray, 50, 150)\n",
                    "\n",
                    "ctrl_bin = (ctrl_gray > 127).astype(np.uint8)\n",
                    "res_bin = (res_edges > 127).astype(np.uint8)\n",
                    "res_dilated = cv2.dilate(res_bin, np.ones((5, 5), np.uint8))\n",
                    "overlap = np.logical_and(ctrl_bin, res_dilated)\n",
                    "f1_preservation = 2 * np.sum(overlap) / (np.sum(ctrl_bin) + np.sum(res_bin) + 1e-6)\n",
                    "ssim_val, _ = ssim(ctrl_gray, res_edges, full=True)\n",
                    "\n",
                    "print(f\"Edge Preservation F1: {f1_preservation:.4f}\")\n",
                    "print(f\"Structural SSIM: {ssim_val:.4f}\")\n",
                    "\n",
                    "# Display side by side\n",
                    "fig, axes = plt.subplots(1, 3, figsize=(18, 6))\n",
                    "axes[0].imshow(control_image, cmap='gray')\n",
                    "axes[0].set_title(\"Input Control Edges\")\n",
                    "axes[0].axis('off')\n",
                    "\n",
                    "axes[1].imshow(res_edges, cmap='gray')\n",
                    "axes[1].set_title(\"Edges Extracted from Render\")\n",
                    "axes[1].axis('off')\n",
                    "\n",
                    "axes[2].imshow(render_result)\n",
                    "axes[2].set_title(\"Photorealistic 2D Scene Render\")\n",
                    "axes[2].axis('off')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.savefig(\"output/comparison.png\", dpi=150)\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 8. Export Results Bundle for Local Web App\n",
                    "\n",
                    "Creates `kaggle_results_bundle.zip` containing:\n",
                    "- `job_result.png`: The photorealistic render\n",
                    "- `metadata.json`: Generation metadata (prompt, seed, steps, device, timings, scores)\n",
                    "\n",
                    "You can download this zip directly from Kaggle and upload it in the local web app with one click!"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "metadata = {\n",
                    "    \"prompt\": full_prompt,\n",
                    "    \"negative_prompt\": NEGATIVE_PROMPT,\n",
                    "    \"steps\": NUM_STEPS,\n",
                    "    \"guidance_scale\": GUIDANCE_SCALE,\n",
                    "    \"controlnet_conditioning_scale\": CONTROLNET_SCALE,\n",
                    "    \"seed\": SEED,\n",
                    "    \"inference_time_seconds\": duration,\n",
                    "    \"device\": f\"cuda ({torch.cuda.get_device_name(0)})\" if torch.cuda.is_available() else \"cpu\",\n",
                    "    \"model_id\": MODEL_ID,\n",
                    "    \"controlnet_id\": CONTROLNET_ID,\n",
                    "    \"edge_preservation_f1\": round(float(f1_preservation), 4),\n",
                    "    \"structural_ssim\": round(float(ssim_val), 4)\n",
                    "}\n",
                    "\n",
                    "with open(\"output/metadata.json\", \"w\") as f:\n",
                    "    json.dump(metadata, f, indent=2)\n",
                    "\n",
                    "bundle_zip = \"kaggle_results_bundle.zip\"\n",
                    "with zipfile.ZipFile(bundle_zip, \"w\", zipfile.ZIP_DEFLATED) as zf:\n",
                    "    zf.write(\"output/job_result.png\", arcname=\"job_result.png\")\n",
                    "    zf.write(\"output/metadata.json\", arcname=\"metadata.json\")\n",
                    "\n",
                    "print(f\"Results bundle created: {bundle_zip} ({os.path.getsize(bundle_zip) / 1024:.1f} KB)\")\n",
                    "print(\"Download this file from the Kaggle file explorer and import into the local app.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 9. (Optional) Launch Temporary Interactive Gradio Web UI"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import gradio as gr\n",
                    "\n",
                    "def gradio_render(input_image, prompt_text, steps, guidance, control_scale, seed):\n",
                    "    if input_image is None:\n",
                    "        return None, \"Please upload a sketch image.\"\n",
                    "    ctrl = prepare_canny_edge_map(input_image)\n",
                    "    gen = torch.manual_seed(int(seed))\n",
                    "    out = pipe(\n",
                    "        prompt=prompt_text,\n",
                    "        negative_prompt=NEGATIVE_PROMPT,\n",
                    "        image=ctrl,\n",
                    "        num_inference_steps=int(steps),\n",
                    "        guidance_scale=float(guidance),\n",
                    "        controlnet_conditioning_scale=float(control_scale),\n",
                    "        generator=gen\n",
                    "    ).images[0]\n",
                    "    return out, \"Render generated successfully!\"\n",
                    "\n",
                    "demo = gr.Interface(\n",
                    "    fn=gradio_render,\n",
                    "    inputs=[\n",
                    "        gr.Image(type=\"pil\", label=\"Sketch Upload\"),\n",
                    "        gr.Textbox(value=full_prompt, label=\"Prompt\"),\n",
                    "        gr.Slider(15, 50, value=25, step=1, label=\"Steps\"),\n",
                    "        gr.Slider(1.0, 15.0, value=7.5, step=0.5, label=\"Guidance\"),\n",
                    "        gr.Slider(0.1, 2.0, value=1.0, step=0.1, label=\"ControlNet Scale\"),\n",
                    "        gr.Number(value=42, label=\"Seed\")\n",
                    "    ],\n",
                    "    outputs=[gr.Image(label=\"Generated Photorealistic Scene\"), gr.Textbox(label=\"Status\")],\n",
                    "    title=\"Sketch-to-Photorealistic Scene Renderer (ControlNet Canny)\"\n",
                    ")\n",
                    "\n",
                    "# demo.launch(share=True) # Uncomment to launch public Gradio link"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 2
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    out_path = Path("kaggle/sketch_to_photorealistic_inference.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {out_path}")

if __name__ == "__main__":
    generate_kaggle_notebook()
