import React from 'react';
import { PromptBuilderOptions } from '@/types';
import { Palette, SunMedium, Camera } from 'lucide-react';

interface PromptControlsProps {
  options: PromptBuilderOptions;
  onChange: (options: PromptBuilderOptions) => void;
}

const MATERIALS_LIST = [
  { id: 'aged_oak', label: 'Aged Oak Wood' },
  { id: 'polished_marble', label: 'Carrara Marble' },
  { id: 'weathered_concrete', label: 'Weathered Concrete' },
  { id: 'brushed_steel', label: 'Brushed Steel' },
  { id: 'brickwork', label: 'Exposed Brick' },
  { id: 'theatrical_velvet', label: 'Crimson Velvet' },
  { id: 'distressed_leather', label: 'Vintage Leather' },
  { id: 'glass_metal', label: 'Plate Glass & Metal' },
];

const LIGHTING_OPTIONS = [
  { id: 'dramatic_stage', label: 'Dramatic Stage Spotlights' },
  { id: 'golden_hour', label: 'Low-Angle Golden Hour' },
  { id: 'moody_chiaroscuro', label: 'Moody Chiaroscuro' },
  { id: 'clean_studio', label: 'Clean 3-Point Studio' },
  { id: 'overcast_diffuse', label: 'Overcast Diffuse Daylight' },
  { id: 'night_noir', label: 'Night Noir & Tungsten' },
];

const STYLE_OPTIONS = [
  { id: 'architectural_photo', label: 'Architectural Photography (Hasselblad)' },
  { id: 'cinematic_still', label: 'Cinematic Film Still (35mm)' },
  { id: 'theatrical_set', label: 'Theatrical Stage Set' },
  { id: 'hyperrealistic_interior', label: 'Photorealistic Interior Design' },
];

export const PromptControls: React.FC<PromptControlsProps> = ({ options, onChange }) => {
  const toggleMaterial = (id: string) => {
    const current = options.materials || [];
    const next = current.includes(id)
      ? current.filter((m) => m !== id)
      : [...current, id];
    onChange({ ...options, materials: next });
  };

  return (
    <div className="space-y-4 pt-2 border-t border-zinc-200 dark:border-zinc-800">
      <div className="flex items-center justify-between">
        <label className="text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
          3. Scene Description & Physical Materials
        </label>
      </div>

      {/* Scene description */}
      <div className="space-y-1.5">
        <div className="flex justify-between items-center text-xs text-zinc-600 dark:text-zinc-400">
          <span className="font-medium">Scene Subject & Set Architecture:</span>
        </div>
        <textarea
          rows={3}
          value={options.scene_description}
          onChange={(e) => onChange({ ...options, scene_description: e.target.value })}
          placeholder="e.g. classical grand theater stage with arch, Roman columns, and steps"
          className="w-full text-xs p-2.5 rounded border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900 text-zinc-900 dark:text-zinc-100 focus:outline-none focus:ring-1 focus:ring-zinc-400 dark:focus:ring-zinc-600 font-sans"
        />
      </div>

      {/* Materials selection */}
      <div className="space-y-1.5">
        <div className="flex items-center gap-1.5 text-xs text-zinc-600 dark:text-zinc-400 font-medium">
          <Palette className="w-3.5 h-3.5" />
          <span>Tactile Materials:</span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {MATERIALS_LIST.map((mat) => {
            const isSelected = options.materials.includes(mat.id);
            return (
              <button
                key={mat.id}
                type="button"
                onClick={() => toggleMaterial(mat.id)}
                className={`text-[11px] px-2.5 py-1 rounded border transition-colors font-mono ${
                  isSelected
                    ? 'border-zinc-900 dark:border-zinc-100 bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900 font-medium'
                    : 'border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/60 text-zinc-600 dark:text-zinc-400 hover:border-zinc-300'
                }`}
              >
                {mat.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Lighting selection */}
      <div className="space-y-1.5">
        <div className="flex items-center gap-1.5 text-xs text-zinc-600 dark:text-zinc-400 font-medium">
          <SunMedium className="w-3.5 h-3.5" />
          <span>Volumetric Lighting:</span>
        </div>
        <select
          value={options.lighting}
          onChange={(e) => onChange({ ...options, lighting: e.target.value })}
          className="w-full text-xs p-2 rounded border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900 text-zinc-900 dark:text-zinc-100 focus:outline-none font-sans"
        >
          {LIGHTING_OPTIONS.map((opt) => (
            <option key={opt.id} value={opt.id}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>

      {/* Render style selection */}
      <div className="space-y-1.5">
        <div className="flex items-center gap-1.5 text-xs text-zinc-600 dark:text-zinc-400 font-medium">
          <Camera className="w-3.5 h-3.5" />
          <span>Photographic Render Style:</span>
        </div>
        <select
          value={options.render_style}
          onChange={(e) => onChange({ ...options, render_style: e.target.value })}
          className="w-full text-xs p-2 rounded border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900 text-zinc-900 dark:text-zinc-100 focus:outline-none font-sans"
        >
          {STYLE_OPTIONS.map((opt) => (
            <option key={opt.id} value={opt.id}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>

      {/* Negative Prompt */}
      <div className="space-y-1">
        <div className="flex justify-between items-center text-[11px] text-zinc-500 font-mono">
          <span>Negative Prompt (Tokens to Suppress):</span>
        </div>
        <input
          type="text"
          value={options.negative_prompt}
          onChange={(e) => onChange({ ...options, negative_prompt: e.target.value })}
          className="w-full text-[11px] p-2 rounded border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900 text-zinc-700 dark:text-zinc-300 font-mono"
        />
      </div>
    </div>
  );
};
