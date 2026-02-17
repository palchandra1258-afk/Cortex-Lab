"use client";

import {
  Sparkles,
  Code2,
  Calculator,
  BookOpen,
  BrainCircuit,
  AlertCircle,
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
    color: "text-violet-400",
    bg: "bg-violet-500/8 border-violet-500/15 hover:border-violet-500/30",
  },
  {
    icon: Code2,
    label: "Code",
    text: "Write a Python function that implements binary search on a sorted list. Include type hints, docstring, and handle edge cases.",
    color: "text-emerald-400",
    bg: "bg-emerald-500/8 border-emerald-500/15 hover:border-emerald-500/30",
  },
  {
    icon: BrainCircuit,
    label: "Reasoning",
    text: "A farmer has 17 sheep. All but 9 die. How many are left? Think through this carefully and explain your reasoning.",
    color: "text-amber-400",
    bg: "bg-amber-500/8 border-amber-500/15 hover:border-amber-500/30",
  },
  {
    icon: BookOpen,
    label: "Explain",
    text: "Explain the concept of Reinforcement Learning from Human Feedback (RLHF) as used in training large language models. Include how it differs from supervised fine-tuning.",
    color: "text-sky-400",
    bg: "bg-sky-500/8 border-sky-500/15 hover:border-sky-500/30",
  },
];

export function EmptyState({ isOnline, onSuggestion }: Props) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] px-4">
      {/* Logo / Title */}
      <div className="mb-8 text-center">
        <div className="inline-flex items-center justify-center h-16 w-16 rounded-2xl bg-deepseek-600/10 border border-deepseek-500/15 mb-5">
          <Sparkles size={28} className="text-deepseek-400" />
        </div>
        <h2 className="text-2xl font-bold gradient-text mb-2">
          DeepSeek R1
        </h2>
        <p className="text-sm text-surface-500 max-w-md">
          14B parameter reasoning model · Distilled from DeepSeek-R1 ·
          Advanced chain-of-thought reasoning
        </p>
      </div>

      {/* Status */}
      {!isOnline && (
        <div className="mb-8 flex items-center gap-2 rounded-xl bg-amber-500/10 border border-amber-500/20 px-4 py-3 text-sm text-amber-400">
          <AlertCircle size={16} />
          <span>
            Model is loading. Start the backend server with{" "}
            <code className="bg-amber-500/10 px-1.5 py-0.5 rounded text-xs font-mono">
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
            className={`group flex flex-col items-start gap-2.5 rounded-xl border p-4 text-left transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed ${s.bg}`}
          >
            <div className="flex items-center gap-2">
              <s.icon size={15} className={s.color} />
              <span className={`text-xs font-semibold ${s.color}`}>
                {s.label}
              </span>
            </div>
            <p className="text-xs leading-relaxed text-surface-400 group-hover:text-surface-300 transition-colors line-clamp-3">
              {s.text}
            </p>
          </button>
        ))}
      </div>

      {/* Tips */}
      <div className="mt-8 text-center space-y-1">
        <p className="text-[11px] text-surface-600">
          💡 Tip: Use temperature 0.6 for best results · For math, ask to put answers in \boxed&#123;&#125;
        </p>
        <p className="text-[11px] text-surface-600">
          The model uses &lt;think&gt; tags to show its reasoning process
        </p>
      </div>
    </div>
  );
}
