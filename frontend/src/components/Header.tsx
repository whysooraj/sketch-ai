import React from 'react';
import { Cpu, Layers } from 'lucide-react';

interface HeaderProps {
  backendHealthy: boolean;
  activeJobId?: string;
}

export const Header: React.FC<HeaderProps> = ({ backendHealthy, activeJobId }) => {
  return (
    <header className="border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 px-6 py-4 sticky top-0 z-30">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900 rounded-md">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-semibold text-lg tracking-tight text-zinc-900 dark:text-zinc-100">
                Sketch-to-Photorealistic Scene Renderer
              </h1>
              <span className="text-xs px-2 py-0.5 rounded bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 font-mono">
                ControlNet SD 1.5
              </span>
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              Spatial preservation & photorealistic material synthesis for set designs
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {activeJobId && (
            <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900 text-xs font-mono text-zinc-600 dark:text-zinc-400">
              <span className="text-zinc-400">Job:</span>
              <span className="font-medium text-zinc-800 dark:text-zinc-200">{activeJobId}</span>
            </div>
          )}

          <div className="flex items-center gap-2 px-2.5 py-1 rounded border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900 text-xs text-zinc-600 dark:text-zinc-400">
            <span
              className={`w-2 h-2 rounded-full ${
                backendHealthy ? 'bg-emerald-500' : 'bg-rose-500'
              }`}
            />
            <span className="font-mono">API: {backendHealthy ? 'Online (8000)' : 'Offline'}</span>
          </div>

          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900 text-xs text-zinc-600 dark:text-zinc-400">
            <Cpu className="w-3.5 h-3.5 text-zinc-500" />
            <span className="font-mono">Local CPU / Kaggle T4</span>
          </div>
        </div>
      </div>
    </header>
  );
};
