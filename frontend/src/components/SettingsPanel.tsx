"use client";

import { ChatSettings } from "@/lib/types";
import { X, RotateCcw, Brain } from "lucide-react";

interface Props {
  settings: ChatSettings;
  onUpdate: (settings: ChatSettings) => void;
  onClose: () => void;
}

export function SettingsPanel({ settings, onUpdate, onClose }: Props) {
  const handleReset = () => {
    onUpdate({
      temperature: 0.6,
      topP: 0.95,
      maxTokens: 2048,
      stream: true,
      useRAG: true,
    });
  };

  return (
    <div className="absolute bottom-24 left-1/2 -translate-x-1/2 z-40 w-full max-w-md">
      <div className="rounded-2xl bg-surface-900/95 border border-surface-700/50 backdrop-blur-xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-surface-800/50">
          <h3 className="text-sm font-semibold text-surface-200">
            Generation Settings
          </h3>
          <div className="flex items-center gap-1">
            <button
              onClick={handleReset}
              className="rounded-lg p-1.5 text-surface-500 hover:text-surface-300 hover:bg-surface-800/50 transition-colors"
              title="Reset to defaults"
            >
              <RotateCcw size={14} />
            </button>
            <button
              onClick={onClose}
              className="rounded-lg p-1.5 text-surface-500 hover:text-surface-300 hover:bg-surface-800/50 transition-colors"
            >
              <X size={14} />
            </button>
          </div>
        </div>

        {/* Controls */}
        <div className="px-5 py-4 space-y-5">
          {/* Temperature */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-medium text-surface-300">
                Temperature
              </label>
              <span className="text-xs font-mono text-deepseek-400 bg-deepseek-500/10 px-2 py-0.5 rounded">
                {settings.temperature.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="2"
              step="0.05"
              value={settings.temperature}
              onChange={(e) =>
                onUpdate({ ...settings, temperature: parseFloat(e.target.value) })
              }
              className="w-full h-1.5 rounded-full appearance-none cursor-pointer bg-surface-700 accent-deepseek-500"
            />
            <div className="flex justify-between mt-1 text-[10px] text-surface-600">
              <span>Precise (0)</span>
              <span className="text-deepseek-500/60">Recommended: 0.6</span>
              <span>Creative (2)</span>
            </div>
          </div>

          {/* Top-P */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-medium text-surface-300">
                Top-P (Nucleus Sampling)
              </label>
              <span className="text-xs font-mono text-deepseek-400 bg-deepseek-500/10 px-2 py-0.5 rounded">
                {settings.topP.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={settings.topP}
              onChange={(e) =>
                onUpdate({ ...settings, topP: parseFloat(e.target.value) })
              }
              className="w-full h-1.5 rounded-full appearance-none cursor-pointer bg-surface-700 accent-deepseek-500"
            />
            <div className="flex justify-between mt-1 text-[10px] text-surface-600">
              <span>Focused (0)</span>
              <span>Diverse (1)</span>
            </div>
          </div>

          {/* Max Tokens */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-medium text-surface-300">
                Max Tokens
              </label>
              <span className="text-xs font-mono text-deepseek-400 bg-deepseek-500/10 px-2 py-0.5 rounded">
                {settings.maxTokens}
              </span>
            </div>
            <input
              type="range"
              min="64"
              max="8192"
              step="64"
              value={settings.maxTokens}
              onChange={(e) =>
                onUpdate({
                  ...settings,
                  maxTokens: parseInt(e.target.value),
                })
              }
              className="w-full h-1.5 rounded-full appearance-none cursor-pointer bg-surface-700 accent-deepseek-500"
            />
            <div className="flex justify-between mt-1 text-[10px] text-surface-600">
              <span>64</span>
              <span>8192</span>
            </div>
          </div>

          {/* Streaming toggle */}
          <div className="flex items-center justify-between py-1">
            <div>
              <label className="text-xs font-medium text-surface-300">
                Streaming
              </label>
              <p className="text-[10px] text-surface-600 mt-0.5">
                Show tokens as they are generated
              </p>
            </div>
            <button
              onClick={() =>
                onUpdate({ ...settings, stream: !settings.stream })
              }
              className={`relative h-6 w-11 rounded-full transition-colors duration-200 ${
                settings.stream
                  ? "bg-deepseek-600"
                  : "bg-surface-700"
              }`}
            >
              <span
                className={`absolute top-1 left-1 h-4 w-4 rounded-full bg-white transition-transform duration-200 shadow-sm ${
                  settings.stream ? "translate-x-5" : "translate-x-0"
                }`}
              />
            </button>
          </div>

          {/* RAG Memory toggle */}
          <div className="flex items-center justify-between py-1">
            <div>
              <label className="text-xs font-medium text-surface-300 flex items-center gap-1.5">
                <Brain size={12} className="text-deepseek-400" />
                RAG Memory
              </label>
              <p className="text-[10px] text-surface-600 mt-0.5">
                Enable retrieval-augmented generation with memory
              </p>
            </div>
            <button
              onClick={() =>
                onUpdate({ ...settings, useRAG: !settings.useRAG })
              }
              className={`relative h-6 w-11 rounded-full transition-colors duration-200 ${
                settings.useRAG
                  ? "bg-emerald-600"
                  : "bg-surface-700"
              }`}
            >
              <span
                className={`absolute top-1 left-1 h-4 w-4 rounded-full bg-white transition-transform duration-200 shadow-sm ${
                  settings.useRAG ? "translate-x-5" : "translate-x-0"
                }`}
              />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
