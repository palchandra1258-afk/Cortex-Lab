"use client";

import {
  PanelLeftClose,
  Plus,
  MessageSquare,
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
    <aside className="sidebar-enter flex w-72 flex-col border-r border-slate-200/80 bg-white relative">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-slate-100">
        <div className="flex items-center gap-2.5">
          <div className="h-8 w-8 rounded-xl flex items-center justify-center">
            <svg width="32" height="32" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect width="48" height="48" rx="12" fill="url(#logo-bg)"/>
              <path d="M24 12C17.373 12 12 17.373 12 24s5.373 12 12 12 12-5.373 12-12S30.627 12 24 12z" fill="none" stroke="white" strokeWidth="1.5" opacity="0.3"/>
              <path d="M24 16c-4.418 0-8 3.582-8 8s3.582 8 8 8 8-3.582 8-8-3.582-8-8-8z" fill="none" stroke="white" strokeWidth="1.5" opacity="0.5"/>
              <circle cx="24" cy="24" r="3.5" fill="white"/>
              <path d="M24 14v-2M24 36v-2M14 24h-2M36 24h-2" stroke="white" strokeWidth="1.5" strokeLinecap="round" opacity="0.4"/>
              <path d="M17.05 17.05l-1.41-1.41M32.36 32.36l-1.41-1.41M17.05 30.95l-1.41 1.41M32.36 15.64l-1.41 1.41" stroke="white" strokeWidth="1.2" strokeLinecap="round" opacity="0.25"/>
              <path d="M20.5 24a3.5 3.5 0 0 1 3.5-3.5" stroke="white" strokeWidth="1" strokeLinecap="round" opacity="0.6"/>
              <defs>
                <linearGradient id="logo-bg" x1="0" y1="0" x2="48" y2="48" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#6366f1"/>
                  <stop offset="1" stopColor="#8b5cf6"/>
                </linearGradient>
              </defs>
            </svg>
          </div>
          <div>
            <span className="text-sm font-semibold text-slate-800 tracking-tight block">
              Cortex Lab
            </span>
            <span className="text-[9px] text-slate-400 font-medium tracking-wider uppercase">
              AI Research
            </span>
          </div>
        </div>
        <button
          onClick={onToggle}
          className="rounded-lg p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-all duration-200"
        >
          <PanelLeftClose size={16} />
        </button>
      </div>

      {/* New Chat Button */}
      <div className="p-3">
        <button
          onClick={onNewChat}
          className="flex w-full items-center gap-2.5 rounded-xl border border-slate-200 px-3.5 py-2.5 text-sm text-slate-600 hover:bg-slate-50 hover:border-slate-300 hover:text-slate-800 transition-all duration-200 group"
        >
          <Plus size={15} className="text-indigo-500 group-hover:text-indigo-600 transition-colors" />
          <span>New Chat</span>
        </button>
      </div>

      {/* Conversation List */}
      <div className="flex-1 overflow-y-auto px-3 py-1">
        <p className="mb-2 px-2 text-[10px] font-medium uppercase tracking-widest text-slate-400">
          Conversations
        </p>
        <div className="space-y-0.5">
          {conversations.map((conv) => (
            <button
              key={conv.id}
              onClick={() => onSelect(conv.id)}
              className={`group flex w-full items-center gap-2.5 rounded-xl px-3 py-2.5 text-left text-sm transition-all duration-200 ${
                conv.id === activeId && activeView === "chat"
                  ? "bg-indigo-50 text-indigo-700 border border-indigo-200 shadow-sm shadow-indigo-100"
                  : "text-slate-500 hover:bg-slate-50 hover:text-slate-700 border border-transparent"
              }`}
            >
              <MessageSquare
                size={14}
                className={
                  conv.id === activeId && activeView === "chat"
                    ? "text-indigo-500"
                    : "text-slate-400 group-hover:text-slate-500"
                }
              />
              <span className="flex-1 truncate">{conv.title}</span>
            </button>
          ))}
        </div>
      </div>

      {/* RAG Navigation */}
      {onNavigate && (
        <div className="border-t border-slate-100 px-3 py-2">
          <p className="mb-2 px-2 text-[10px] font-medium uppercase tracking-widest text-slate-400">
            RAG System
          </p>
          <div className="space-y-0.5">
            {navItems.map(({ view, icon: Icon, label }) => (
              <button
                key={view}
                onClick={() => onNavigate(view)}
                className={`flex w-full items-center gap-2.5 rounded-xl px-3 py-2.5 text-left text-sm transition-all duration-200 ${
                  activeView === view
                    ? "bg-indigo-50 text-indigo-700 border border-indigo-200 shadow-sm shadow-indigo-100"
                    : "text-slate-500 hover:bg-slate-50 hover:text-slate-700 border border-transparent"
                }`}
              >
                <Icon
                  size={14}
                  className={
                    activeView === view
                      ? "text-indigo-500"
                      : "text-slate-400"
                  }
                />
                <span>{label}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="border-t border-slate-100 p-4">
        <div className="rounded-xl bg-slate-50 border border-slate-200 px-3.5 py-3">
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-indigo-500 animate-glow-pulse" />
            <p className="text-[10px] font-medium text-slate-400 uppercase tracking-wider">Model</p>
          </div>
          <p className="text-xs text-slate-700 mt-1.5 font-medium">
            DeepSeek-R1-7B
          </p>
          <p className="text-[9px] text-slate-400 mt-0.5">
            Curriculum Fine-tuned · 15 Stages
          </p>
        </div>
      </div>
    </aside>
  );
}
