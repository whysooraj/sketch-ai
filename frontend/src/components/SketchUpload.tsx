import React, { useRef, useState } from 'react';
import { UploadCloud, Sparkles, AlertCircle } from 'lucide-react';

interface SketchUploadProps {
  onUploadSuccess: (file: File) => Promise<void>;
  onLoadSample: () => Promise<void>;
  isUploading: boolean;
  rawPreviewUrl?: string;
}

export const SketchUpload: React.FC<SketchUploadProps> = ({
  onUploadSuccess,
  onLoadSample,
  isUploading,
  rawPreviewUrl,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    setError(null);
    if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) {
      setError('Please upload a valid PNG, JPEG, or WebP image');
      return;
    }
    if (file.size > 15 * 1024 * 1024) {
      setError('File size must be under 15MB');
      return;
    }
    try {
      await onUploadSuccess(file);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Upload failed');
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <label className="text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
          1. Hand-Drawn Sketch Input
        </label>
        <button
          type="button"
          onClick={onLoadSample}
          disabled={isUploading}
          className="text-xs font-medium text-zinc-700 dark:text-zinc-300 hover:text-zinc-950 dark:hover:text-white flex items-center gap-1.5 transition-colors disabled:opacity-50"
        >
          <Sparkles className="w-3.5 h-3.5" />
          <span>Load Academic Sample</span>
        </button>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`relative border border-dashed rounded-lg p-5 flex flex-col items-center justify-center text-center cursor-pointer transition-all ${
          isDragging
            ? 'border-zinc-900 dark:border-zinc-100 bg-zinc-100 dark:bg-zinc-800/50'
            : 'border-zinc-300 dark:border-zinc-700 hover:border-zinc-400 dark:hover:border-zinc-600 bg-zinc-50/50 dark:bg-zinc-900/40'
        } ${isUploading ? 'opacity-60 pointer-events-none' : ''}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              handleFile(e.target.files[0]);
            }
          }}
        />

        <div className="p-2.5 rounded-full bg-zinc-200/70 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300 mb-2">
          <UploadCloud className="w-5 h-5" />
        </div>
        <p className="text-xs font-medium text-zinc-900 dark:text-zinc-100 mb-0.5">
          Drop architectural set sketch here, or <span className="underline">browse</span>
        </p>
        <p className="text-[11px] text-zinc-500 dark:text-zinc-400">
          Supports PNG, JPEG, WebP (Max 15MB)
        </p>
      </div>

      {error && (
        <div className="flex items-center gap-2 p-2.5 rounded border border-rose-200 dark:border-rose-900/50 bg-rose-50 dark:bg-rose-950/30 text-rose-700 dark:text-rose-400 text-xs">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {rawPreviewUrl && (
        <div className="relative rounded-lg overflow-hidden border border-zinc-200 dark:border-zinc-800 bg-zinc-100 dark:bg-zinc-900 p-2 flex items-center justify-center">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={rawPreviewUrl}
            alt="Uploaded Sketch Preview"
            className="max-h-48 w-auto object-contain rounded"
          />
        </div>
      )}
    </div>
  );
};
