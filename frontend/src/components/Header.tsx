"use client";

import { PanelLeftOpen, Cpu, Loader2, WifiOff, CheckCircle2, FlaskConical } from "lucide-react";
import { ModelStatus } from "@/lib/types";

interface Props {
  modelStatus: ModelStatus;
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
}

export function Header({ modelStatus, sidebarOpen, onToggleSidebar }: Props) {
  const statusColor = modelStatus.model_loaded
    ? "text-emerald-600"
    : modelStatus.status === "loading"
      ? "text-amber-600"
      : "text-red-500";

  const statusBg = modelStatus.model_loaded
    ? "bg-emerald-50 border-emerald-200"
    : modelStatus.status === "loading"
      ? "bg-amber-50 border-amber-200"
      : "bg-red-50 border-red-200";

  const statusDot = modelStatus.model_loaded
    ? "bg-emerald-500"
    : modelStatus.status === "loading"
      ? "bg-amber-500"
      : "bg-red-500";

  const StatusIcon = modelStatus.model_loaded
    ? CheckCircle2
    : modelStatus.status === "loading"
      ? Loader2
      : WifiOff;

  const isFT = modelStatus.model_info.fine_tuned;
  const stages = modelStatus.model_info.training_stages_completed ?? 0;

  return (
    <header className="relative flex items-center justify-between border-b border-slate-200/80 bg-white/90 backdrop-blur-2xl px-5 py-3.5">
      {/* Subtle top highlight */}
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-slate-200 to-transparent" />

      <div className="flex items-center gap-3">
        {!sidebarOpen && (
          <button
            onClick={onToggleSidebar}
            className="rounded-lg p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-all duration-200"
          >
            <PanelLeftOpen size={18} />
          </button>
        )}
        <div className="flex items-center gap-2.5">
          <h1 className="text-sm font-semibold text-slate-700 tracking-tight">
            Cortex Lab
          </h1>
          <span className="rounded-md bg-indigo-50 border border-indigo-200 px-1.5 py-0.5 text-[10px] font-semibold text-indigo-600 tracking-wide">
            7B
          </span>
          {isFT && (
            <span className="flex items-center gap-1 rounded-md bg-violet-50 border border-violet-200 px-1.5 py-0.5 text-[10px] font-semibold text-violet-600 tracking-wide">
              <FlaskConical size={9} />
              Fine-Tuned · {stages}/15
            </span>
          )}
        </div>
      </div>

      {/* Status indicator */}
      <div
        className={`flex items-center gap-2.5 rounded-xl border px-3.5 py-2 transition-all duration-300 ${statusBg}`}
      >
        <div className="relative">
          <div className={`h-2 w-2 rounded-full ${statusDot}`} />
          {modelStatus.model_loaded && (
            <div className={`absolute inset-0 h-2 w-2 rounded-full ${statusDot} animate-ping opacity-40`} />
          )}
        </div>
        <span className={`text-[11px] font-medium ${statusColor}`}>
          {modelStatus.model_loaded
            ? "Online"
            : modelStatus.status === "loading"
              ? "Loading Model…"
              : "Offline"}
        </span>
        {modelStatus.model_info.quantization && (
          <span className="text-[10px] text-slate-400 border-l border-slate-200 pl-2.5 ml-0.5">
            {modelStatus.model_info.quantization}
          </span>
        )}
        {modelStatus.model_info.device && (
          <span className="flex items-center gap-1 text-[10px] text-slate-400 border-l border-slate-200 pl-2.5 ml-0.5">
            <Cpu size={10} />
            {modelStatus.model_info.device.split(" ").slice(-1)[0]}
          </span>
        )}
      </div>
    </header>
  );
}
