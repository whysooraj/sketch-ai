# AGENTS.md — Sketch-to-Photorealistic Renderer

## Project objective

Build an academic application that transforms hand-drawn set-design sketches into photorealistic 2D scene renders.

The system must preserve the spatial structure and major objects in the original sketch while generating realistic materials, textures, lighting and shadows.

## Hard constraints

1. Total budget: ₹0.
2. Assume no dedicated GPU on the development laptop.
3. Never purchase or activate paid cloud services.
4. Never enable paid billing or use paid APIs.
5. Use free pretrained models.
6. Never train a diffusion model from scratch.
7. Use Kaggle free GPU sessions for heavy inference.
8. All frontend and backend development must work locally.
9. Never assume free GPU availability is guaranteed.
10. Provide a CPU-compatible fallback for preprocessing and testing.

## Technology stack

- Python
- FastAPI
- OpenCV
- Pillow
- PyTorch
- Hugging Face Diffusers
- Stable Diffusion 1.5
- ControlNet Canny
- Next.js
- TypeScript
- Tailwind CSS
- Pytest

## AI architecture

Implement a modular pipeline:

1. Accept a hand-drawn sketch.
2. Validate image format and dimensions.
3. Convert the image to grayscale.
4. Remove noise.
5. Extract edges using Canny.
6. Normalize and resize the sketch.
7. Build a scene description from user input.
8. Pass the processed sketch to ControlNet.
9. Generate a photorealistic image using a pretrained diffusion model.
10. Export the final image and generation metadata.

Use a pretrained Stable Diffusion 1.5 + ControlNet Canny model as the initial baseline.

Keep the model inference code separate from the API and frontend.

## GPU strategy

Create a standalone Kaggle notebook.

The notebook must:
- Install required free dependencies.
- Load pretrained model weights.
- Accept uploaded sketches.
- Execute inference on an available GPU.
- Allow prompt and generation parameter changes.
- Save output images.
- Export generation metadata.

Do not depend on a permanently running cloud server.

The application must support manual image transfer between the local application and Kaggle.

An optional temporary Gradio integration may be implemented later.

## Frontend requirements

Build a clean, modern interface with:

- Image upload.
- Original sketch preview.
- Processed sketch preview.
- Scene description input.
- Material selection.
- Lighting selection.
- Render style selection.
- Generation settings.
- Generated image preview.
- Before/after comparison.
- Image download.

## Backend requirements

Implement:

- Image upload endpoint.
- Image preprocessing endpoint.
- Job creation and status endpoints.
- Result upload endpoint.
- Result retrieval endpoint.
- Input validation.
- Error handling.
- Local file storage.

## Evaluation

Measure:

- Edge preservation.
- Structural similarity where appropriate.
- Generation time.
- Visual quality.
- Consistency between sketch and generated scene.

Do not fabricate evaluation results.

## Execution workflow

Work in milestones.

Before implementation:
1. Inspect the environment.
2. Check installed software and available hardware.
3. Create a detailed implementation plan.
4. Identify dependencies and model licensing requirements.

During implementation:
1. Complete one milestone at a time.
2. Run tests after each milestone.
3. Fix errors before proceeding.
4. Keep README and technical documentation updated.
5. Avoid unnecessary dependencies.
6. Do not create fake or placeholder functionality and claim it works.

At completion:
1. Run the full test suite.
2. Verify the local application.
3. Verify the Kaggle notebook instructions.
4. Document setup and execution.
5. List known limitations.
6. Provide a final project status report.
