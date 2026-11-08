'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Header } from '@/components/Header';
import { SketchUpload } from '@/components/SketchUpload';
import { PreprocessingControls } from '@/components/PreprocessingControls';
import { PromptControls } from '@/components/PromptControls';
import { InferenceControls } from '@/components/InferenceControls';
import { ComparisonSlider } from '@/components/ComparisonSlider';
import { EvaluationView } from '@/components/EvaluationView';
import {
  Job,
  PreprocessRequest,
  PreprocessResult,
  PromptBuilderOptions,
  InferenceRequest,
} from '@/types';
import { Eye, Layers, Split, History } from 'lucide-react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Home() {
  const [backendHealthy, setBackendHealthy] = useState<boolean>(false);
  const [activeJob, setActiveJob] = useState<Job | null>(null);
  const [allJobs, setAllJobs] = useState<Job[]>([]);
  const [activeTab, setActiveTab] = useState<'slider' | 'gallery' | 'eval'>('slider');

  // Loading states
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [isPreprocessing, setIsPreprocessing] = useState<boolean>(false);
  const [isInferencing, setIsInferencing] = useState<boolean>(false);
  const [isEvaluating, setIsEvaluating] = useState<boolean>(false);
  const [globalError, setGlobalError] = useState<string | null>(null);

  // Preprocessing config
  const [preprocessConfig, setPreprocessConfig] = useState<PreprocessRequest>({
    low_threshold: 100,
    high_threshold: 200,
    blur_kernel: 3,
    denoise_strength: 5,
    invert_input: false,
    auto_detect_inversion: true,
    target_size: 512,
  });
  const [preprocessResult, setPreprocessResult] = useState<PreprocessResult | undefined>();

  // Prompt options
  const [promptOptions, setPromptOptions] = useState<PromptBuilderOptions>({
    scene_description: 'classical grand theater stage with arch, Greek stone columns, and steps',
    materials: ['polished_marble', 'aged_oak'],
    lighting: 'dramatic_stage',
    render_style: 'theatrical_set',
    negative_prompt:
      'blurry, low quality, worst quality, cartoon, drawing, painting, sketch, distorted geometry, watermark',
  });

  // Inference config
  const [inferenceConfig, setInferenceConfig] = useState<InferenceRequest>({
    prompt_options: promptOptions,
    num_inference_steps: 30,
    guidance_scale: 7.5,
    controlnet_conditioning_scale: 1.0,
    seed: 42,
    force_fallback: true,
  });

  // Sync prompt options into inference config
  useEffect(() => {
    setInferenceConfig((prev) => ({
      ...prev,
      prompt_options: promptOptions,
    }));
  }, [promptOptions]);

  // Health check & jobs list
  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        setBackendHealthy(true);
      } else {
        setBackendHealthy(false);
      }
    } catch {
      setBackendHealthy(false);
    }
  }, []);

  const fetchJobs = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/jobs`);
      if (res.ok) {
        const jobs: Job[] = await res.json();
        setAllJobs(jobs);
      }
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    checkHealth();
    fetchJobs();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, [checkHealth, fetchJobs]);

  // Upload sketch
  const handleUpload = async (file: File) => {
    setGlobalError(null);
    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`${API_BASE}/api/upload`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Upload failed');
      }
      const newJob: Job = await res.json();
      setActiveJob(newJob);
      setPreprocessResult(undefined);
      fetchJobs();
    } catch (e: unknown) {
      setGlobalError(e instanceof Error ? e.message : 'Upload failed');
      throw e;
    } finally {
      setIsUploading(false);
    }
  };

  // Load sample sketch
  const handleLoadSample = async () => {
    setGlobalError(null);
    setIsUploading(true);
    try {
      // Fetch sample from backend storage
      const res = await fetch(`${API_BASE}/storage/uploads/sample_stage_sketch.png`);
      if (!res.ok) throw new Error('Sample sketch not found on backend');
      const blob = await res.blob();
      const file = new File([blob], 'sample_stage_sketch.png', { type: 'image/png' });
      await handleUpload(file);
    } catch (e: unknown) {
      setGlobalError(e instanceof Error ? e.message : 'Failed to load sample');
    } finally {
      setIsUploading(false);
    }
  };

  // Run preprocessing
  const handlePreprocess = async () => {
    if (!activeJob) return;
    setGlobalError(null);
    setIsPreprocessing(true);
    try {
      const res = await fetch(`${API_BASE}/api/preprocess/${activeJob.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(preprocessConfig),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Preprocessing failed');
      }
      const result: PreprocessResult = await res.json();
      setPreprocessResult(result);
      // Refresh active job
      const jobRes = await fetch(`${API_BASE}/api/jobs/${activeJob.id}`);
      if (jobRes.ok) {
        setActiveJob(await jobRes.json());
      }
      fetchJobs();
    } catch (e: unknown) {
      setGlobalError(e instanceof Error ? e.message : 'Preprocessing failed');
    } finally {
      setIsPreprocessing(false);
    }
  };

  // Run local inference
  const handleRunInference = async () => {
    if (!activeJob) return;
    setGlobalError(null);
    setIsInferencing(true);
    try {
      const res = await fetch(`${API_BASE}/api/inference/${activeJob.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(inferenceConfig),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Inference failed');
      }
      const updatedJob: Job = await res.json();
      setActiveJob(updatedJob);
      setActiveTab('slider');
      fetchJobs();

      // Automatically compute evaluation metrics
      handleRunEval(updatedJob.id);
    } catch (e: unknown) {
      setGlobalError(e instanceof Error ? e.message : 'Inference failed');
    } finally {
      setIsInferencing(false);
    }
  };

  // Export Kaggle bundle
  const handleExportKaggleBundle = () => {
    if (!activeJob) return;
    window.open(`${API_BASE}/api/jobs/${activeJob.id}/export-bundle`, '_blank');
  };

  // Import Kaggle result
  const handleImportKaggleResult = async (file: File) => {
    if (!activeJob) return;
    setGlobalError(null);
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/api/jobs/${activeJob.id}/import-result`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Import failed');
    }
    const updatedJob: Job = await res.json();
    setActiveJob(updatedJob);
    setActiveTab('slider');
    fetchJobs();
    handleRunEval(updatedJob.id);
  };

  // Run evaluation
  const handleRunEval = async (targetJobId?: string) => {
    const jid = targetJobId || activeJob?.id;
    if (!jid) return;
    setIsEvaluating(true);
    try {
      const res = await fetch(`${API_BASE}/api/eval/${jid}?tolerance=2`, {
        method: 'POST',
      });
      if (res.ok) {
        const jobRes = await fetch(`${API_BASE}/api/jobs/${jid}`);
        if (jobRes.ok) {
          setActiveJob(await jobRes.json());
        }
      }
    } catch {
      // ignore
    } finally {
      setIsEvaluating(false);
    }
  };

  const rawSketchUrl = activeJob?.original_image_url
    ? `${API_BASE}${activeJob.original_image_url}`
    : undefined;
  const edgeSketchUrl = activeJob?.preprocessed_image_url
    ? `${API_BASE}${activeJob.preprocessed_image_url}`
    : undefined;
  const resultImageUrl = activeJob?.result_image_url
    ? `${API_BASE}${activeJob.result_image_url}`
    : undefined;

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 flex flex-col font-sans">
      <Header backendHealthy={backendHealthy} activeJobId={activeJob?.id} />

      {/* Global alert error if any */}
      {globalError && (
        <div className="max-w-7xl mx-auto px-6 pt-4 w-full">
          <div className="p-3 rounded-lg border border-rose-200 dark:border-rose-900/50 bg-rose-50 dark:bg-rose-950/40 text-xs text-rose-700 dark:text-rose-400">
            {globalError}
          </div>
        </div>
      )}

      {/* Main workspace layout */}
      <main className="max-w-7xl mx-auto px-6 py-6 w-full flex-1 grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Side: Pipeline Controls (5 cols) */}
        <section className="lg:col-span-5 bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-5 space-y-6 shadow-xs">
          <SketchUpload
            onUploadSuccess={handleUpload}
            onLoadSample={handleLoadSample}
            isUploading={isUploading}
            rawPreviewUrl={rawSketchUrl}
          />

          <PreprocessingControls
            config={preprocessConfig}
            onChange={setPreprocessConfig}
            onRunPreprocess={handlePreprocess}
            isProcessing={isPreprocessing}
            result={preprocessResult}
            hasUploadedImage={!!activeJob}
          />

          <PromptControls options={promptOptions} onChange={setPromptOptions} />

          <InferenceControls
            request={inferenceConfig}
            onChange={setInferenceConfig}
            onRunLocalInference={handleRunInference}
            onExportKaggleBundle={handleExportKaggleBundle}
            onImportKaggleResult={handleImportKaggleResult}
            isInferencing={isInferencing}
            isPreprocessed={!!activeJob?.preprocessed_image_url}
            jobId={activeJob?.id}
            hasResult={!!activeJob?.result_image_url}
          />
        </section>

        {/* Right Side: Visualizer, Comparison & Metrics (7 cols) */}
        <section className="lg:col-span-7 bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-5 space-y-5 shadow-xs sticky top-24">
          {/* Tabs bar */}
          <div className="flex items-center justify-between border-b border-zinc-200 dark:border-zinc-800 pb-3">
            <div className="flex items-center space-x-1">
              <button
                type="button"
                onClick={() => setActiveTab('slider')}
                className={`text-xs px-3 py-1.5 rounded font-medium transition-colors flex items-center gap-1.5 ${
                  activeTab === 'slider'
                    ? 'bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900'
                    : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200'
                }`}
              >
                <Split className="w-3.5 h-3.5" />
                <span>Before / After Split</span>
              </button>

              <button
                type="button"
                onClick={() => setActiveTab('gallery')}
                className={`text-xs px-3 py-1.5 rounded font-medium transition-colors flex items-center gap-1.5 ${
                  activeTab === 'gallery'
                    ? 'bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900'
                    : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200'
                }`}
              >
                <Layers className="w-3.5 h-3.5" />
                <span>Side-by-Side Gallery</span>
              </button>

              <button
                type="button"
                onClick={() => setActiveTab('eval')}
                className={`text-xs px-3 py-1.5 rounded font-medium transition-colors flex items-center gap-1.5 ${
                  activeTab === 'eval'
                    ? 'bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900'
                    : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200'
                }`}
              >
                <Eye className="w-3.5 h-3.5" />
                <span>Evaluation & Metrics</span>
              </button>
            </div>

            {activeJob && (
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400">
                Status: {activeJob.status}
              </span>
            )}
          </div>

          {/* Tab 1: Interactive Comparison Slider */}
          {activeTab === 'slider' && (
            <div>
              {resultImageUrl && rawSketchUrl ? (
                <ComparisonSlider
                  originalUrl={rawSketchUrl}
                  edgeUrl={edgeSketchUrl}
                  resultUrl={resultImageUrl}
                />
              ) : (
                <div className="aspect-square max-h-[500px] w-full border border-dashed border-zinc-200 dark:border-zinc-800 rounded-lg flex flex-col items-center justify-center p-8 text-center bg-zinc-50/50 dark:bg-zinc-900/30">
                  <Split className="w-8 h-8 text-zinc-400 mb-2" />
                  <p className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
                    Before / After comparison will activate once a scene is rendered
                  </p>
                  <p className="text-[11px] text-zinc-400 mt-1 max-w-sm">
                    Upload a sketch, extract Canny edges, and click &ldquo;Render Scene&rdquo; or import
                    results from Kaggle.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Tab 2: Side-by-Side Gallery */}
          {activeTab === 'gallery' && (
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div className="space-y-1.5">
                <span className="text-[11px] font-mono text-zinc-500 uppercase block">
                  1. Input Sketch
                </span>
                <div className="aspect-square rounded border border-zinc-200 dark:border-zinc-800 bg-zinc-100 dark:bg-zinc-950 flex items-center justify-center overflow-hidden">
                  {rawSketchUrl ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={rawSketchUrl}
                      alt="Input Sketch"
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <span className="text-xs text-zinc-400">Pending upload</span>
                  )}
                </div>
              </div>

              <div className="space-y-1.5">
                <span className="text-[11px] font-mono text-zinc-500 uppercase block">
                  2. Canny Edge Map
                </span>
                <div className="aspect-square rounded border border-zinc-200 dark:border-zinc-800 bg-zinc-950 flex items-center justify-center overflow-hidden">
                  {edgeSketchUrl ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={edgeSketchUrl}
                      alt="Canny Edges"
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <span className="text-xs text-zinc-400">Pending extraction</span>
                  )}
                </div>
              </div>

              <div className="space-y-1.5">
                <span className="text-[11px] font-mono text-zinc-500 uppercase block">
                  3. Photorealistic Render
                </span>
                <div className="aspect-square rounded border border-zinc-200 dark:border-zinc-800 bg-zinc-100 dark:bg-zinc-950 flex items-center justify-center overflow-hidden">
                  {resultImageUrl ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={resultImageUrl}
                      alt="Rendered Scene"
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <span className="text-xs text-zinc-400">Pending inference</span>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Tab 3: Evaluation & Metrics */}
          {activeTab === 'eval' && (
            <EvaluationView
              metrics={activeJob?.evaluation}
              metadata={activeJob?.generation_metadata}
              onRunEval={() => handleRunEval()}
              isEvaluating={isEvaluating}
              resultImageUrl={resultImageUrl}
              jobId={activeJob?.id}
            />
          )}

          {/* Recent Jobs History Footer */}
          {allJobs.length > 0 && (
            <div className="pt-4 border-t border-zinc-200 dark:border-zinc-800">
              <div className="flex items-center gap-1.5 text-xs text-zinc-500 font-mono mb-2">
                <History className="w-3.5 h-3.5" />
                <span>Recent Jobs:</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {allJobs.slice(0, 5).map((job) => (
                  <button
                    key={job.id}
                    onClick={() => setActiveJob(job)}
                    className={`text-[11px] font-mono px-2 py-1 rounded border transition-colors ${
                      activeJob?.id === job.id
                        ? 'border-zinc-800 dark:border-zinc-200 bg-zinc-100 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100'
                        : 'border-zinc-200 dark:border-zinc-800 hover:border-zinc-400 text-zinc-500'
                    }`}
                  >
                    {job.id} ({job.status})
                  </button>
                ))}
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
