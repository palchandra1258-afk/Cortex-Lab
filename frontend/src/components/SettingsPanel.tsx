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
    <div className="absolute bottom-24 left-1/2 -translate-x-1/2 z-40 w-full max-w-md fade-in">
      <div className="rounded-2xl bg-white border border-slate-200 backdrop-blur-2xl shadow-2xl shadow-slate-200/60 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-slate-100">
          <h3 className="text-sm font-semibold text-slate-700">
            Generation Settings
          </h3>
          <div className="flex items-center gap-1">
            <button
              onClick={handleReset}
              className="rounded-lg p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-all duration-200"
              title="Reset to defaults"
            >
              <RotateCcw size={14} />
            </button>
            <button
              onClick={onClose}
              className="rounded-lg p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-all duration-200"
            >
              <X size={14} />
            </button>
          </div>
        </div>

        {/* Controls */}
        <div className="px-5 py-4 space-y-5">
          {/* Temperature */}
          <div>
            <div className="flex items-center justify-between mb-2.5">
              <label className="text-xs font-medium text-slate-600">
                Temperature
              </label>
              <span className="text-xs font-mono text-indigo-600 bg-indigo-50 border border-indigo-200 px-2 py-0.5 rounded-md">
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
              className="w-full"
            />
            <div className="flex justify-between mt-1.5 text-[10px] text-slate-400">
              <span>Precise (0)</span>
              <span className="text-indigo-500">Recommended: 0.6</span>
              <span>Creative (2)</span>
            </div>
          </div>

          {/* Top-P */}
          <div>
            <div className="flex items-center justify-between mb-2.5">
              <label className="text-xs font-medium text-slate-600">
                Top-P (Nucleus Sampling)
              </label>
              <span className="text-xs font-mono text-indigo-600 bg-indigo-50 border border-indigo-200 px-2 py-0.5 rounded-md">
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
              className="w-full"
            />
            <div className="flex justify-between mt-1.5 text-[10px] text-slate-400">
              <span>Focused (0)</span>
              <span>Diverse (1)</span>
            </div>
          </div>

          {/* Max Tokens */}
          <div>
            <div className="flex items-center justify-between mb-2.5">
              <label className="text-xs font-medium text-slate-600">
                Max Tokens
              </label>
              <span className="text-xs font-mono text-indigo-600 bg-indigo-50 border border-indigo-200 px-2 py-0.5 rounded-md">
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
              className="w-full"
            />
            <div className="flex justify-between mt-1.5 text-[10px] text-slate-400">
              <span>64</span>
              <span>8192</span>
            </div>
          </div>

          {/* Streaming toggle */}
          <div className="flex items-center justify-between py-1">
            <div>
              <label className="text-xs font-medium text-slate-600">
                Streaming
              </label>
              <p className="text-[10px] text-slate-400 mt-0.5">
                Show tokens as they are generated
              </p>
            </div>
            <button
              onClick={() =>
                onUpdate({ ...settings, stream: !settings.stream })
              }
              className={`relative h-6 w-11 rounded-full transition-all duration-300 ${
                settings.stream
                  ? "bg-indigo-600 shadow-sm shadow-indigo-200"
                  : "bg-slate-300"
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
              <label className="text-xs font-medium text-slate-600 flex items-center gap-1.5">
                <Brain size={12} className="text-indigo-500" />
                RAG Memory
              </label>
              <p className="text-[10px] text-slate-400 mt-0.5">
                Enable retrieval-augmented generation with memory
              </p>
            </div>
            <button
              onClick={() =>
                onUpdate({ ...settings, useRAG: !settings.useRAG })
              }
              className={`relative h-6 w-11 rounded-full transition-all duration-300 ${
                settings.useRAG
                  ? "bg-emerald-500 shadow-sm shadow-emerald-200"
                  : "bg-slate-300"
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
