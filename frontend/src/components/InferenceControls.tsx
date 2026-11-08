import React, { useRef, useState } from 'react';
import { InferenceRequest } from '@/types';
import { Play, Download, Upload, Cloud, CheckCircle, AlertCircle } from 'lucide-react';

interface InferenceControlsProps {
  request: InferenceRequest;
  onChange: (req: InferenceRequest) => void;
  onRunLocalInference: () => Promise<void>;
  onExportKaggleBundle: () => void;
  onImportKaggleResult: (file: File) => Promise<void>;
  isInferencing: boolean;
  isPreprocessed: boolean;
  jobId?: string;
  hasResult: boolean;
}

export const InferenceControls: React.FC<InferenceControlsProps> = ({
  request,
  onChange,
  onRunLocalInference,
  onExportKaggleBundle,
  onImportKaggleResult,
  isInferencing,
  isPreprocessed,
  jobId,
  hasResult,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [importing, setImporting] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);

  const handleZipFile = async (file: File) => {
    setImportError(null);
    if (!file.name.endsWith('.zip')) {
      setImportError('Please select a valid .zip bundle from Kaggle');
      return;
    }
    setImporting(true);
    try {
      await onImportKaggleResult(file);
    } catch (e: unknown) {
      setImportError(e instanceof Error ? e.message : 'Import failed');
    } finally {
      setImporting(false);
    }
  };

  return (
    <div className="space-y-4 pt-2 border-t border-zinc-200 dark:border-zinc-800">
      <div className="flex items-center justify-between">
        <label className="text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
          4. Diffusion Generation & Compute Options
        </label>
        {hasResult && (
          <span className="inline-flex items-center gap-1 text-[11px] font-mono text-emerald-600 dark:text-emerald-400">
            <CheckCircle className="w-3.5 h-3.5" />
            Render Ready
          </span>
        )}
      </div>

      {/* Hyperparameters */}
      <div className="grid grid-cols-2 gap-3 text-xs">
        <div className="space-y-1">
          <div className="flex justify-between text-zinc-600 dark:text-zinc-400 font-mono">
            <span>Steps:</span>
            <span>{request.num_inference_steps}</span>
          </div>
          <input
            type="range"
            min="15"
            max="50"
            step="1"
            value={request.num_inference_steps}
            onChange={(e) => onChange({ ...request, num_inference_steps: parseInt(e.target.value) })}
            className="w-full accent-zinc-800 dark:accent-zinc-200"
          />
        </div>

        <div className="space-y-1">
          <div className="flex justify-between text-zinc-600 dark:text-zinc-400 font-mono">
            <span>CFG Guidance:</span>
            <span>{request.guidance_scale}</span>
          </div>
          <input
            type="range"
            min="3.0"
            max="15.0"
            step="0.5"
            value={request.guidance_scale}
            onChange={(e) => onChange({ ...request, guidance_scale: parseFloat(e.target.value) })}
            className="w-full accent-zinc-800 dark:accent-zinc-200"
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 text-xs">
        <div className="space-y-1">
          <div className="flex justify-between text-zinc-600 dark:text-zinc-400 font-mono">
            <span>ControlNet Scale:</span>
            <span>{request.controlnet_conditioning_scale}</span>
          </div>
          <input
            type="range"
            min="0.2"
            max="1.8"
            step="0.1"
            value={request.controlnet_conditioning_scale}
            onChange={(e) =>
              onChange({ ...request, controlnet_conditioning_scale: parseFloat(e.target.value) })
            }
            className="w-full accent-zinc-800 dark:accent-zinc-200"
          />
        </div>

        <div className="space-y-1">
          <div className="flex justify-between text-zinc-600 dark:text-zinc-400 font-mono">
            <span>Random Seed:</span>
          </div>
          <input
            type="number"
            value={request.seed}
            onChange={(e) => onChange({ ...request, seed: parseInt(e.target.value) || 42 })}
            className="w-full bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded px-2 py-1 text-xs text-zinc-800 dark:text-zinc-200 font-mono"
          />
        </div>
      </div>

      {/* Primary Local CPU Inference */}
      <button
        type="button"
        onClick={onRunLocalInference}
        disabled={!isPreprocessed || isInferencing}
        className="w-full py-2.5 px-3 rounded bg-zinc-900 hover:bg-zinc-800 text-white dark:bg-zinc-100 dark:hover:bg-white dark:text-zinc-950 font-medium text-xs flex items-center justify-center gap-2 transition-colors disabled:opacity-40 disabled:cursor-not-allowed shadow-xs"
      >
        <Play className={`w-3.5 h-3.5 fill-current ${isInferencing ? 'animate-pulse' : ''}`} />
        <span>{isInferencing ? 'Rendering Scene...' : 'Render Scene (Local Engine)'}</span>
      </button>

      {/* Kaggle Zero-Cost Free GPU Bridge */}
      <div className="pt-2 border-t border-dashed border-zinc-200 dark:border-zinc-800 space-y-2">
        <div className="flex items-center justify-between text-[11px] text-zinc-500 font-mono">
          <span className="flex items-center gap-1">
            <Cloud className="w-3.5 h-3.5" />
            <span>Kaggle Free Cloud GPU (T4 / P100):</span>
          </span>
          <span className="text-emerald-600 dark:text-emerald-400 font-semibold">₹0 Cost</span>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <button
            type="button"
            onClick={onExportKaggleBundle}
            disabled={!isPreprocessed || !jobId}
            className="py-1.5 px-2.5 rounded border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900 hover:bg-zinc-100 dark:hover:bg-zinc-800 text-zinc-700 dark:text-zinc-300 font-medium text-[11px] flex items-center justify-center gap-1.5 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export Bundle</span>
          </button>

          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={!jobId || importing}
            className="py-1.5 px-2.5 rounded border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900 hover:bg-zinc-100 dark:hover:bg-zinc-800 text-zinc-700 dark:text-zinc-300 font-medium text-[11px] flex items-center justify-center gap-1.5 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <Upload className="w-3.5 h-3.5" />
            <span>{importing ? 'Importing...' : 'Import Result'}</span>
          </button>

          <input
            ref={fileInputRef}
            type="file"
            accept=".zip"
            className="hidden"
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                handleZipFile(e.target.files[0]);
              }
            }}
          />
        </div>

        {importError && (
          <div className="flex items-center gap-1.5 p-2 rounded bg-rose-50 dark:bg-rose-950/30 text-rose-700 dark:text-rose-400 text-[11px]">
            <AlertCircle className="w-3.5 h-3.5 shrink-0" />
            <span>{importError}</span>
          </div>
        )}
      </div>
    </div>
  );
};
