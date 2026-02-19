"use client";

import {
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  MessageSquare,
  Trash2,
  Brain,
  Network,
  BarChart3,
} from "lucide-react";

interface Conversation {
  id: string;
  title: string;
  date: string;
}

type ActiveView = "chat" | "memories" | "graph" | "dashboard";

interface Props {
  open: boolean;
  conversations: Conversation[];
  activeId: string;
  onSelect: (id: string) => void;
  onNewChat: () => void;
  onToggle: () => void;
  activeView?: ActiveView;
  onNavigate?: (view: ActiveView) => void;
}

export function Sidebar({
  open,
  conversations,
  activeId,
  onSelect,
  onNewChat,
  onToggle,
  activeView = "chat",
  onNavigate,
}: Props) {
  if (!open) return null;

  const navItems: { view: ActiveView; icon: typeof Brain; label: string }[] = [
    { view: "memories", icon: Brain, label: "Memory Browser" },
    { view: "graph", icon: Network, label: "Knowledge Graph" },
    { view: "dashboard", icon: BarChart3, label: "RAG Dashboard" },
  ];

  return (
    <aside className="flex w-72 flex-col border-r border-surface-800/50 bg-surface-950 relative">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-surface-800/40">
        <div className="flex items-center gap-2">
          <div className="h-7 w-7 rounded-lg bg-deepseek-600 flex items-center justify-center">
            <span className="text-xs font-bold text-white">CL</span>
          </div>
          <span className="text-sm font-semibold text-surface-200">
            Cortex Lab
          </span>
        </div>
        <button
          onClick={onToggle}
          className="rounded-lg p-1.5 text-surface-500 hover:text-surface-300 hover:bg-surface-800/50 transition-colors"
        >
          <PanelLeftClose size={16} />
        </button>
      </div>

      {/* New Chat Button */}
      <div className="p-3">
        <button
          onClick={onNewChat}
          className="flex w-full items-center gap-2.5 rounded-xl border border-surface-700/40 px-3.5 py-2.5 text-sm text-surface-300 hover:bg-surface-800/40 hover:border-surface-600/50 transition-all duration-200"
        >
          <Plus size={15} />
          <span>New Chat</span>
        </button>
      </div>

      {/* Conversation List */}
      <div className="flex-1 overflow-y-auto px-3 py-1">
        <p className="mb-2 px-2 text-[10px] font-medium uppercase tracking-wider text-surface-600">
          Conversations
        </p>
        <div className="space-y-0.5">
          {conversations.map((conv) => (
            <button
              key={conv.id}
              onClick={() => onSelect(conv.id)}
              className={`group flex w-full items-center gap-2.5 rounded-lg px-3 py-2.5 text-left text-sm transition-all duration-150 ${
                conv.id === activeId && activeView === "chat"
                  ? "bg-deepseek-600/10 text-deepseek-300 border border-deepseek-500/15"
                  : "text-surface-400 hover:bg-surface-800/40 hover:text-surface-200 border border-transparent"
              }`}
            >
              <MessageSquare
                size={14}
                className={
                  conv.id === activeId && activeView === "chat"
                    ? "text-deepseek-400"
                    : "text-surface-600"
                }
              />
              <span className="flex-1 truncate">{conv.title}</span>
            </button>
          ))}
        </div>
      </div>

      {/* RAG Navigation */}
      {onNavigate && (
        <div className="border-t border-surface-800/40 px-3 py-2">
          <p className="mb-2 px-2 text-[10px] font-medium uppercase tracking-wider text-surface-600">
            RAG System
          </p>
          <div className="space-y-0.5">
            {navItems.map(({ view, icon: Icon, label }) => (
              <button
                key={view}
                onClick={() => onNavigate(view)}
                className={`flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-sm transition-all duration-150 ${
                  activeView === view
                    ? "bg-deepseek-600/10 text-deepseek-300 border border-deepseek-500/15"
                    : "text-surface-400 hover:bg-surface-800/40 hover:text-surface-200 border border-transparent"
                }`}
              >
                <Icon
                  size={14}
                  className={
                    activeView === view
                      ? "text-deepseek-400"
                      : "text-surface-600"
                  }
                />
                <span>{label}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="border-t border-surface-800/40 p-4">
        <div className="rounded-lg bg-surface-900/40 px-3 py-2.5">
          <p className="text-[10px] font-medium text-surface-500">Model</p>
          <p className="text-xs text-surface-300 mt-0.5">
            DeepSeek-R1-1.5B
          </p>
        </div>
      </div>
    </aside>
  );
}
