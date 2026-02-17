"use client";

import { PanelLeftOpen, Cpu, Loader2, WifiOff, CheckCircle2 } from "lucide-react";
import { ModelStatus } from "@/lib/types";

interface Props {
  modelStatus: ModelStatus;
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
}

export function Header({ modelStatus, sidebarOpen, onToggleSidebar }: Props) {
  const statusColor = modelStatus.model_loaded
    ? "text-emerald-400"
    : modelStatus.status === "loading"
      ? "text-amber-400"
      : "text-red-400";

  const statusBg = modelStatus.model_loaded
    ? "bg-emerald-500/10 border-emerald-500/20"
    : modelStatus.status === "loading"
      ? "bg-amber-500/10 border-amber-500/20"
      : "bg-red-500/10 border-red-500/20";

  const StatusIcon = modelStatus.model_loaded
    ? CheckCircle2
    : modelStatus.status === "loading"
      ? Loader2
      : WifiOff;

  return (
    <header className="flex items-center justify-between border-b border-surface-800/50 bg-surface-950/80 backdrop-blur-xl px-4 py-3">
      <div className="flex items-center gap-3">
        {!sidebarOpen && (
          <button
            onClick={onToggleSidebar}
            className="rounded-lg p-1.5 text-surface-500 hover:text-surface-300 hover:bg-surface-800/50 transition-colors"
          >
            <PanelLeftOpen size={18} />
          </button>
        )}
        <div className="flex items-center gap-2">
          <h1 className="text-sm font-semibold text-surface-200">
            DeepSeek R1 Chat
          </h1>
          <span className="rounded-md bg-deepseek-600/15 px-1.5 py-0.5 text-[10px] font-medium text-deepseek-400">
            14B
          </span>
        </div>
      </div>

      {/* Status indicator */}
      <div
        className={`flex items-center gap-2 rounded-lg border px-3 py-1.5 ${statusBg}`}
      >
        <StatusIcon
          size={13}
          className={`${statusColor} ${
            modelStatus.status === "loading" ? "animate-spin" : ""
          }`}
        />
        <span className={`text-[11px] font-medium ${statusColor}`}>
          {modelStatus.model_loaded
            ? "Online"
            : modelStatus.status === "loading"
              ? "Loading Model…"
              : "Offline"}
        </span>
        {modelStatus.model_info.quantization && (
          <span className="text-[10px] text-surface-500 border-l border-surface-700/50 pl-2 ml-0.5">
            {modelStatus.model_info.quantization}
          </span>
        )}
        {modelStatus.model_info.device && (
          <span className="flex items-center gap-1 text-[10px] text-surface-500 border-l border-surface-700/50 pl-2 ml-0.5">
            <Cpu size={10} />
            {modelStatus.model_info.device.split(" ").slice(-1)[0]}
          </span>
        )}
      </div>
    </header>
  );
}
