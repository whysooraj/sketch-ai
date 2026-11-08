# Quantitative Evaluation Report: Sketch-to-Photorealistic Scene Renderer

This benchmark evaluates edge preservation, structural similarity, and generation latency.

## Benchmark Results

| Test Scenario | Edge Preservation (F1) | IoU | Structural SSIM | Inference Time | Total Pipeline Time |
|---|---|---|---|---|---|
| Classical Theater (Oak + Marble, Dramatic Stage) | 79.48% | 65.95% | 0.8164 | 0.118s | 0.171s |
| Ancient Temple Ruins (Stone + Daylight) | 75.07% | 60.09% | 0.8166 | 0.128s | 0.17s |
| Moody Opera Stage (Crimson Velvet + Chiaroscuro) | 82.26% | 69.87% | 0.8135 | 0.119s | 0.161s |

## Metric Definitions

1. **Edge Preservation (F1 Score)**: Evaluates Dice similarity between input sketch Canny edge maps and edges extracted from rendered scenes, allowing a 2-pixel tolerance radius for brush stroke variance.
2. **Edge IoU**: Intersection over Union between the sketch edge contour mask and the render edge contour mask.
3. **Structural Similarity (SSIM)**: Mean Structural Similarity Index computed between the normalized structural channels.
4. **Inference Latency**: Duration of diffusion conditioning and image rendering.
