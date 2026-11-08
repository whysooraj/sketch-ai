import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Sliders } from 'lucide-react';

interface ComparisonSliderProps {
  originalUrl: string;
  edgeUrl?: string;
  resultUrl: string;
  aspectRatio?: string;
}

export const ComparisonSlider: React.FC<ComparisonSliderProps> = ({
  originalUrl,
  edgeUrl,
  resultUrl,
}) => {
  const [sliderPosition, setSliderPosition] = useState<number>(50);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [leftSource, setLeftSource] = useState<'original' | 'edge'>('original');
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const position = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(position);
  }, []);

  const handleTouchMove = useCallback((e: TouchEvent) => {
    if (!isDragging) return;
    handleMove(e.touches[0].clientX);
  }, [isDragging, handleMove]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging) return;
    handleMove(e.clientX);
  }, [isDragging, handleMove]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      window.addEventListener('touchmove', handleTouchMove);
      window.addEventListener('touchend', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('touchend', handleMouseUp);
    };
  }, [isDragging, handleMouseMove, handleTouchMove, handleMouseUp]);

  const activeLeftUrl = leftSource === 'edge' && edgeUrl ? edgeUrl : originalUrl;

  return (
    <div className="flex flex-col space-y-3">
      <div className="flex items-center justify-between text-xs text-zinc-500 dark:text-zinc-400">
        <div className="flex items-center gap-2">
          <span>Compare input:</span>
          <div className="inline-flex rounded border border-zinc-200 dark:border-zinc-800 p-0.5 bg-zinc-100 dark:bg-zinc-900">
            <button
              onClick={() => setLeftSource('original')}
              className={`px-2 py-0.5 rounded text-xs font-medium transition-colors ${
                leftSource === 'original'
                  ? 'bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 shadow-xs'
                  : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-200'
              }`}
            >
              Raw Sketch
            </button>
            {edgeUrl && (
              <button
                onClick={() => setLeftSource('edge')}
                className={`px-2 py-0.5 rounded text-xs font-medium transition-colors ${
                  leftSource === 'edge'
                    ? 'bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 shadow-xs'
                    : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-200'
                }`}
              >
                Canny Edges
              </button>
            )}
          </div>
        </div>

        <div className="flex items-center gap-4">
          <span className="font-mono">Left: {leftSource === 'original' ? 'Sketch' : 'Edges'}</span>
          <span className="font-mono">Right: Photorealistic Render</span>
        </div>
      </div>

      <div
        ref={containerRef}
        className="relative w-full aspect-square max-h-[580px] select-none overflow-hidden rounded-lg border border-zinc-200 dark:border-zinc-800 bg-zinc-900 shadow-xs cursor-ew-resize"
        onMouseDown={(e) => {
          setIsDragging(true);
          handleMove(e.clientX);
        }}
        onTouchStart={(e) => {
          setIsDragging(true);
          handleMove(e.touches[0].clientX);
        }}
      >
        {/* Right Image (Generated Photorealistic Render) */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={resultUrl}
          alt="Generated Photorealistic Render"
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
        />

        {/* Left Image (Original Sketch or Preprocessed Canny) clipped */}
        <div
          className="absolute inset-0 overflow-hidden pointer-events-none"
          style={{ width: `${sliderPosition}%` }}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={activeLeftUrl}
            alt="Source Sketch or Edges"
            className="absolute inset-0 w-full h-full object-contain pointer-events-none"
            style={{ width: containerRef.current ? `${containerRef.current.clientWidth}px` : '100%', maxWidth: 'none' }}
          />
        </div>

        {/* Divider Bar */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-white shadow-md pointer-events-none"
          style={{ left: `${sliderPosition}%` }}
        >
          <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-zinc-700 shadow-md flex items-center justify-center text-zinc-700 dark:text-zinc-200">
            <Sliders className="w-3.5 h-3.5 rotate-90" />
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between text-xs text-zinc-400 font-mono">
        <span>Position: {Math.round(sliderPosition)}%</span>
        <span>Drag center divider horizontally</span>
      </div>
    </div>
  );
};
