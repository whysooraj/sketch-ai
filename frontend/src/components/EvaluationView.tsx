import React from 'react';
import { EvaluationMetrics, GenerationMetadata } from '@/types';
import { Download, BarChart2 } from 'lucide-react';

interface EvaluationViewProps {
  metrics?: EvaluationMetrics;
  metadata?: GenerationMetadata;
  onRunEval: () => Promise<void>;
  isEvaluating: boolean;
  resultImageUrl?: string;
  jobId?: string;
}

export const EvaluationView: React.FC<EvaluationViewProps> = ({
  metrics,
  metadata,
  onRunEval,
  isEvaluating,
  resultImageUrl,
  jobId,
}) => {
  const downloadResult = () => {
    if (!resultImageUrl) return;
    const link = document.createElement('a');
    link.href = resultImageUrl;
    link.download = `photorealistic_render_${jobId || 'scene'}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const downloadMetadata = () => {
    if (!metadata) return;
    const blob = new Blob([JSON.stringify({ metadata, metrics }, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `generation_metadata_${jobId || 'scene'}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-4">
      {/* Top action bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-zinc-200 dark:border-zinc-800">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400 flex items-center gap-1.5">
          <BarChart2 className="w-4 h-4" />
          <span>Quantitative Metrics & Generation Details</span>
        </h3>

        <div className="flex items-center gap-2">
          {!metrics && (
            <button
              type="button"
              onClick={onRunEval}
              disabled={isEvaluating}
              className="text-xs px-2.5 py-1 rounded bg-zinc-900 hover:bg-zinc-800 text-white dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white font-medium flex items-center gap-1.5 transition-colors disabled:opacity-50"
            >
              <span>{isEvaluating ? 'Evaluating...' : 'Compute Metrics'}</span>
            </button>
          )}

          {resultImageUrl && (
            <button
              type="button"
              onClick={downloadResult}
              className="text-xs px-2.5 py-1 rounded border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 hover:bg-zinc-50 dark:hover:bg-zinc-800 text-zinc-800 dark:text-zinc-200 font-medium flex items-center gap-1.5 transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download Render</span>
            </button>
          )}

          {metadata && (
            <button
              type="button"
              onClick={downloadMetadata}
              className="text-xs px-2.5 py-1 rounded border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 hover:bg-zinc-50 dark:hover:bg-zinc-800 text-zinc-800 dark:text-zinc-200 font-medium flex items-center gap-1.5 transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Metadata JSON</span>
            </button>
          )}
        </div>
      </div>

      {/* Metrics Cards */}
      {metrics ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="p-3 rounded-lg border border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/40">
            <span className="text-[11px] font-mono text-zinc-500 uppercase block mb-1">
              Edge F1-Score
            </span>
            <div className="text-xl font-bold font-mono tracking-tight text-zinc-900 dark:text-zinc-100 tabular-nums">
              {(metrics.edge_preservation_f1 * 100).toFixed(1)}%
            </div>
            <span className="text-[10px] text-zinc-400">Dice similarity</span>
          </div>

          <div className="p-3 rounded-lg border border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/40">
            <span className="text-[11px] font-mono text-zinc-500 uppercase block mb-1">
              Edge IoU
            </span>
            <div className="text-xl font-bold font-mono tracking-tight text-zinc-900 dark:text-zinc-100 tabular-nums">
              {(metrics.edge_preservation_iou * 100).toFixed(1)}%
            </div>
            <span className="text-[10px] text-zinc-400">Intersection over Union</span>
          </div>

          <div className="p-3 rounded-lg border border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/40">
            <span className="text-[11px] font-mono text-zinc-500 uppercase block mb-1">
              Structural SSIM
            </span>
            <div className="text-xl font-bold font-mono tracking-tight text-zinc-900 dark:text-zinc-100 tabular-nums">
              {metrics.structural_similarity_ssim.toFixed(3)}
            </div>
            <span className="text-[10px] text-zinc-400">Geometry consistency</span>
          </div>

          <div className="p-3 rounded-lg border border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/40">
            <span className="text-[11px] font-mono text-zinc-500 uppercase block mb-1">
              Eval Latency
            </span>
            <div className="text-xl font-bold font-mono tracking-tight text-zinc-900 dark:text-zinc-100 tabular-nums">
              {metrics.evaluation_time_seconds}s
            </div>
            <span className="text-[10px] text-zinc-400">Edge & SSIM analysis</span>
          </div>
        </div>
      ) : (
        <div className="p-4 rounded border border-dashed border-zinc-200 dark:border-zinc-800 text-center text-xs text-zinc-500">
          Click &ldquo;Compute Metrics&rdquo; to evaluate edge preservation and structural similarity between original sketch and generated render.
        </div>
      )}

      {/* Generation Metadata Details */}
      {metadata && (
        <div className="p-4 rounded-lg border border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/30 space-y-2.5 text-xs">
          <div className="font-semibold text-zinc-800 dark:text-zinc-200">
            Generation Parameters
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-[11px]">
            <div>
              <span className="text-zinc-400 block">Device:</span>
              <span className="text-zinc-800 dark:text-zinc-200">{metadata.device}</span>
            </div>
            <div>
              <span className="text-zinc-400 block">Duration:</span>
              <span className="text-zinc-800 dark:text-zinc-200 tabular-nums">
                {metadata.inference_time_seconds}s
              </span>
            </div>
            <div>
              <span className="text-zinc-400 block">Steps / CFG:</span>
              <span className="text-zinc-800 dark:text-zinc-200 tabular-nums">
                {metadata.steps} / {metadata.guidance_scale}
              </span>
            </div>
            <div>
              <span className="text-zinc-400 block">Seed:</span>
              <span className="text-zinc-800 dark:text-zinc-200 tabular-nums">
                {metadata.seed}
              </span>
            </div>
          </div>

          <div className="space-y-1">
            <span className="text-zinc-400 font-mono text-[11px] block">Full Prompt:</span>
            <p className="p-2 rounded bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-[11px] text-zinc-700 dark:text-zinc-300 font-sans leading-relaxed">
              {metadata.prompt}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
