import React from 'react';
import { RefreshCw, CheckCircle2 } from 'lucide-react';
import { PreprocessRequest, PreprocessResult } from '@/types';

interface PreprocessingControlsProps {
  config: PreprocessRequest;
  onChange: (newConfig: PreprocessRequest) => void;
  onRunPreprocess: () => Promise<void>;
  isProcessing: boolean;
  result?: PreprocessResult;
  hasUploadedImage: boolean;
}

export const PreprocessingControls: React.FC<PreprocessingControlsProps> = ({
  config,
  onChange,
  onRunPreprocess,
  isProcessing,
  result,
  hasUploadedImage,
}) => {
  return (
    <div className="space-y-4 pt-2 border-t border-zinc-200 dark:border-zinc-800">
      <div className="flex items-center justify-between">
        <label className="text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
          2. Edge Extraction (Canny Hysteresis)
        </label>
        {result && (
          <span className="inline-flex items-center gap-1 text-[11px] font-mono text-emerald-600 dark:text-emerald-400">
            <CheckCircle2 className="w-3.5 h-3.5" />
            Processed (512x512)
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3 text-xs">
        <div className="space-y-1.5">
          <div className="flex justify-between text-zinc-600 dark:text-zinc-400 font-mono">
            <span>Low Threshold:</span>
            <span>{config.low_threshold}</span>
          </div>
          <input
            type="range"
            min="10"
            max="150"
            step="5"
            value={config.low_threshold}
            onChange={(e) => onChange({ ...config, low_threshold: parseInt(e.target.value) })}
            className="w-full accent-zinc-800 dark:accent-zinc-200"
          />
        </div>

        <div className="space-y-1.5">
          <div className="flex justify-between text-zinc-600 dark:text-zinc-400 font-mono">
            <span>High Threshold:</span>
            <span>{config.high_threshold}</span>
          </div>
          <input
            type="range"
            min="100"
            max="250"
            step="5"
            value={config.high_threshold}
            onChange={(e) => onChange({ ...config, high_threshold: parseInt(e.target.value) })}
            className="w-full accent-zinc-800 dark:accent-zinc-200"
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 text-xs">
        <div className="space-y-1.5">
          <div className="flex justify-between text-zinc-600 dark:text-zinc-400 font-mono">
            <span>Denoise Filter:</span>
            <span>{config.denoise_strength}</span>
          </div>
          <input
            type="range"
            min="0"
            max="15"
            step="1"
            value={config.denoise_strength}
            onChange={(e) => onChange({ ...config, denoise_strength: parseInt(e.target.value) })}
            className="w-full accent-zinc-800 dark:accent-zinc-200"
          />
        </div>

        <div className="space-y-1.5">
          <div className="flex justify-between text-zinc-600 dark:text-zinc-400 font-mono">
            <span>Gaussian Blur:</span>
            <span>{config.blur_kernel}px</span>
          </div>
          <select
            value={config.blur_kernel}
            onChange={(e) => onChange({ ...config, blur_kernel: parseInt(e.target.value) })}
            className="w-full bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded px-2 py-1 text-xs text-zinc-800 dark:text-zinc-200 font-mono"
          >
            <option value="1">1 (None)</option>
            <option value="3">3 (Standard)</option>
            <option value="5">5 (Smooth)</option>
          </select>
        </div>
      </div>

      <button
        type="button"
        onClick={onRunPreprocess}
        disabled={!hasUploadedImage || isProcessing}
        className="w-full py-2 px-3 rounded bg-zinc-900 hover:bg-zinc-800 text-white dark:bg-zinc-100 dark:hover:bg-white dark:text-zinc-950 font-medium text-xs flex items-center justify-center gap-2 transition-colors disabled:opacity-40 disabled:cursor-not-allowed shadow-xs"
      >
        <RefreshCw className={`w-3.5 h-3.5 ${isProcessing ? 'animate-spin' : ''}`} />
        <span>{isProcessing ? 'Extracting Edges...' : 'Generate ControlNet Edge Map'}</span>
      </button>

      {result && (
        <div className="p-2.5 rounded bg-zinc-50 dark:bg-zinc-900/60 border border-zinc-200 dark:border-zinc-800 text-[11px] font-mono text-zinc-600 dark:text-zinc-400 grid grid-cols-3 gap-2">
          <div>
            <span className="text-zinc-400 dark:text-zinc-500 block">Edge Pixels:</span>
            <span className="font-semibold text-zinc-800 dark:text-zinc-200 tabular-nums">
              {result.edge_pixel_count.toLocaleString()}
            </span>
          </div>
          <div>
            <span className="text-zinc-400 dark:text-zinc-500 block">Density:</span>
            <span className="font-semibold text-zinc-800 dark:text-zinc-200 tabular-nums">
              {(result.edge_density * 100).toFixed(2)}%
            </span>
          </div>
          <div>
            <span className="text-zinc-400 dark:text-zinc-500 block">Geometry:</span>
            <span className="font-semibold text-zinc-800 dark:text-zinc-200">
              512x512
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
