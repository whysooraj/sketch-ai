export type JobStatus = 'CREATED' | 'PREPROCESSED' | 'PENDING_GPU' | 'COMPLETED' | 'FAILED';

export interface PreprocessRequest {
  low_threshold: number;
  high_threshold: number;
  blur_kernel: number;
  denoise_strength: number;
  invert_input: boolean;
  auto_detect_inversion: boolean;
  target_size: number;
}

export interface PreprocessResult {
  job_id: string;
  original_url: string;
  preprocessed_url: string;
  width: number;
  height: number;
  edge_pixel_count: number;
  edge_density: number;
  parameters: PreprocessRequest;
}

export interface PromptBuilderOptions {
  scene_description: string;
  materials: string[];
  lighting: string;
  render_style: string;
  negative_prompt?: string;
  custom_modifiers?: string;
}

export interface InferenceRequest {
  prompt_options: PromptBuilderOptions;
  num_inference_steps: number;
  guidance_scale: number;
  controlnet_conditioning_scale: number;
  seed: number;
  force_fallback: boolean;
}

export interface GenerationMetadata {
  prompt: string;
  negative_prompt: string;
  steps: number;
  guidance_scale: number;
  controlnet_conditioning_scale: number;
  seed: number;
  inference_time_seconds: number;
  device: string;
  model_id: string;
  controlnet_id: string;
}

export interface EvaluationMetrics {
  edge_preservation_f1: number;
  edge_preservation_iou: number;
  structural_similarity_ssim: number;
  evaluation_time_seconds: number;
  details: {
    sketch_edge_pixels?: number;
    render_edge_pixels?: number;
    tolerance_radius_pixels?: number;
    precision?: number;
    recall?: number;
  };
}

export interface Job {
  id: string;
  status: JobStatus;
  created_at: string;
  updated_at: string;
  original_image_path?: string;
  original_image_url?: string;
  preprocessed_image_path?: string;
  preprocessed_image_url?: string;
  result_image_path?: string;
  result_image_url?: string;
  preprocess_config?: Record<string, unknown>;
  prompt_config?: Record<string, unknown>;
  generation_metadata?: GenerationMetadata;
  evaluation?: EvaluationMetrics;
  error_message?: string;
}
