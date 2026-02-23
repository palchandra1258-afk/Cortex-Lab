"use client";

import {
  Code2,
  Calculator,
  BookOpen,
  BrainCircuit,
  AlertCircle,
  Zap,
  ArrowRight,
} from "lucide-react";

interface Props {
  isOnline: boolean;
  onSuggestion: (text: string) => void;
}

const suggestions = [
  {
    icon: Calculator,
    label: "Math",
    text: "Solve: If f(x) = x³ - 6x² + 11x - 6, find all the roots. Please reason step by step, and put your final answer within \\boxed{}.",
    color: "text-violet-600",
    bg: "bg-violet-50 border-violet-200 hover:border-violet-300 hover:bg-violet-100/70",
    glow: "group-hover:shadow-violet-200/50",
  },
  {
    icon: Code2,
    label: "Code",
    text: "Write a Python function that implements binary search on a sorted list. Include type hints, docstring, and handle edge cases.",
    color: "text-emerald-600",
    bg: "bg-emerald-50 border-emerald-200 hover:border-emerald-300 hover:bg-emerald-100/70",
    glow: "group-hover:shadow-emerald-200/50",
  },
  {
    icon: BrainCircuit,
    label: "Reasoning",
    text: "A farmer has 17 sheep. All but 9 die. How many are left? Think through this carefully and explain your reasoning.",
    color: "text-amber-600",
    bg: "bg-amber-50 border-amber-200 hover:border-amber-300 hover:bg-amber-100/70",
    glow: "group-hover:shadow-amber-200/50",
  },
  {
    icon: BookOpen,
    label: "Explain",
    text: "Explain the concept of Reinforcement Learning from Human Feedback (RLHF) as used in training large language models. Include how it differs from supervised fine-tuning.",
    color: "text-sky-600",
    bg: "bg-sky-50 border-sky-200 hover:border-sky-300 hover:bg-sky-100/70",
    glow: "group-hover:shadow-sky-200/50",
  },
];

export function EmptyState({ isOnline, onSuggestion }: Props) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] px-4 fade-in">
      {/* Hero */}
      <div className="mb-10 text-center relative">
        {/* Gradient orb behind logo */}
        <div className="gradient-orb absolute -inset-20 pointer-events-none" />

        <div className="relative inline-flex items-center justify-center h-20 w-20 rounded-3xl bg-gradient-to-br from-indigo-500 to-violet-600 border border-indigo-300/30 mb-6 shadow-2xl shadow-indigo-200/40">
          <svg width="40" height="40" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M24 12C17.373 12 12 17.373 12 24s5.373 12 12 12 12-5.373 12-12S30.627 12 24 12z" fill="none" stroke="white" strokeWidth="1.5" opacity="0.3"/>
            <path d="M24 16c-4.418 0-8 3.582-8 8s3.582 8 8 8 8-3.582 8-8-3.582-8-8-8z" fill="none" stroke="white" strokeWidth="1.5" opacity="0.5"/>
            <circle cx="24" cy="24" r="3.5" fill="white"/>
            <path d="M24 14v-2M24 36v-2M14 24h-2M36 24h-2" stroke="white" strokeWidth="1.5" strokeLinecap="round" opacity="0.4"/>
            <path d="M17.05 17.05l-1.41-1.41M32.36 32.36l-1.41-1.41M17.05 30.95l-1.41 1.41M32.36 15.64l-1.41 1.41" stroke="white" strokeWidth="1.2" strokeLinecap="round" opacity="0.25"/>
            <path d="M20.5 24a3.5 3.5 0 0 1 3.5-3.5" stroke="white" strokeWidth="1" strokeLinecap="round" opacity="0.6"/>
          </svg>
          <div className="absolute -inset-px rounded-3xl bg-gradient-to-b from-white/20 to-transparent" />
        </div>

        <h2 className="text-3xl font-bold gradient-text mb-3 tracking-tight">
          Cortex Lab
        </h2>
        <p className="text-sm text-slate-500 max-w-md leading-relaxed">
          <span className="text-slate-700 font-medium">DeepSeek-R1-7B</span> reasoning model ·
          Curriculum fine-tuned across 15 stages ·
          Advanced chain-of-thought reasoning
        </p>

        {/* Feature pills */}
        <div className="flex items-center justify-center gap-2 mt-4">
          {["Chain of Thought", "RAG Memory", "Knowledge Graph"].map((feat) => (
            <span
              key={feat}
              className="px-2.5 py-1 rounded-full bg-slate-100 border border-slate-200 text-[10px] text-slate-500 font-medium"
            >
              {feat}
            </span>
          ))}
        </div>
      </div>

      {/* Status */}
      {!isOnline && (
        <div className="mb-8 flex items-center gap-2.5 rounded-xl bg-amber-50 border border-amber-200 px-4 py-3 text-sm text-amber-700 backdrop-blur-sm">
          <AlertCircle size={16} />
          <span>
            Model is loading. Start the backend with{" "}
            <code className="bg-amber-100 px-1.5 py-0.5 rounded-md text-xs font-mono border border-amber-200">
              python server.py
            </code>
          </span>
        </div>
      )}

      {/* Suggestion Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-2xl w-full">
        {suggestions.map((s) => (
          <button
            key={s.label}
            onClick={() => onSuggestion(s.text)}
            disabled={!isOnline}
            className={`group flex flex-col items-start gap-3 rounded-2xl border p-4.5 text-left transition-all duration-300 disabled:opacity-30 disabled:cursor-not-allowed card-hover ${s.bg} ${s.glow}`}
          >
            <div className="flex items-center gap-2">
              <div className={`p-1.5 rounded-lg bg-white/60 ${s.color}`}>
                <s.icon size={14} />
              </div>
              <span className={`text-xs font-semibold tracking-wide ${s.color}`}>
                {s.label}
              </span>
              <ArrowRight
                size={12}
                className="ml-auto text-slate-300 group-hover:text-slate-500 group-hover:translate-x-0.5 transition-all"
              />
            </div>
            <p className="text-xs leading-relaxed text-slate-500 group-hover:text-slate-700 transition-colors line-clamp-3">
              {s.text}
            </p>
          </button>
        ))}
      </div>

      {/* Tips */}
      <div className="mt-10 text-center space-y-1.5">
        <div className="flex items-center justify-center gap-1.5 text-[11px] text-slate-400">
          <Zap size={10} className="text-indigo-500/60" />
          <span>Temperature 0.6 recommended · Ask to put answers in \boxed&#123;&#125; for math</span>
        </div>
        <p className="text-[11px] text-slate-400">
          The model uses &lt;think&gt; tags to show its reasoning process
        </p>
      </div>
    </div>
  );
}
