"use client";

import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { ChatMessage } from "@/lib/types";
import {
  User,
  Bot,
  ChevronDown,
  ChevronRight,
  Brain,
  Copy,
  Check,
  Clock,
  Zap,
  Sparkles,
  FileText,
  Network,
  Shield,
  Timer,
} from "lucide-react";

interface Props {
  message: ChatMessage;
}

export function MessageBubble({ message }: Props) {
  const [showThinking, setShowThinking] = useState(false);
  const [copied, setCopied] = useState(false);
  const thinkingEndRef = useRef<HTMLDivElement>(null);

  const isUser = message.role === "user";
  const hasThinking = !!message.thinking;

  // Parse thinking from streamed content
  let displayContent = message.content;
  let streamedThinking: string | null = null;
  let isCurrentlyThinking = false;

  if (!isUser && message.content.includes("<think>")) {
    const thinkStart = message.content.indexOf("<think>") + 7;
    const thinkEnd = message.content.indexOf("</think>");
    if (thinkEnd > -1) {
      streamedThinking = message.content.slice(thinkStart, thinkEnd).trim();
      displayContent = message.content.slice(thinkEnd + 8).trim();
    } else {
      // Still thinking
      streamedThinking = message.content.slice(thinkStart).trim();
      displayContent = "";
      isCurrentlyThinking = message.isStreaming || false;
    }
  }

  const thinking = message.thinking || streamedThinking;
  const hasOutput = displayContent.trim().length > 0;

  // Auto-expand thinking when streaming
  useEffect(() => {
    if (isCurrentlyThinking && thinking) {
      setShowThinking(true);
    }
  }, [isCurrentlyThinking, thinking]);

  // Auto-scroll thinking panel
  useEffect(() => {
    if (showThinking && isCurrentlyThinking) {
      thinkingEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [thinking, showThinking, isCurrentlyThinking]);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(displayContent || message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={`group py-5 ${isUser ? "" : ""}`}>
      <div className="flex gap-4">
        {/* Avatar */}
        <div className="flex-shrink-0 mt-0.5">
          {isUser ? (
            <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-indigo-50 text-indigo-600 ring-1 ring-indigo-200">
              <User size={15} />
            </div>
          ) : (
            <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-50 to-emerald-100 text-emerald-600 ring-1 ring-emerald-200">
              <Bot size={15} />
            </div>
          )}
        </div>

        {/* Content */}
        <div className="min-w-0 flex-1 space-y-2.5">
          {/* Role label */}
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-slate-700">
              {isUser ? "You" : "Cortex Lab"}
            </span>
            {message.isStreaming && (
              <span className="flex items-center gap-1.5 text-[10px] text-indigo-500">
                <span className="relative flex h-1.5 w-1.5">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-indigo-400 opacity-75" />
                  <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-indigo-400" />
                </span>
                generating
              </span>
            )}
          </div>

          {/* Live Thinking Panel - Enhanced */}
          {thinking && (
            <div className={`rounded-2xl overflow-hidden border transition-all duration-300 ${
              isCurrentlyThinking 
                ? 'border-indigo-200 bg-indigo-50/50 shadow-lg shadow-indigo-100/30' 
                : 'border-slate-200 bg-slate-50/50'
            }`}>
              <button
                onClick={() => setShowThinking((p) => !p)}
                className={`flex w-full items-center gap-2.5 px-4 py-3 text-sm font-medium transition-colors ${
                  isCurrentlyThinking
                    ? 'text-indigo-600 hover:bg-indigo-50'
                    : 'text-slate-500 hover:bg-slate-50'
                }`}
              >
                {isCurrentlyThinking ? (
                  <Brain size={14} className="text-indigo-500 animate-pulse" />
                ) : (
                  <Brain size={14} className="text-slate-400" />
                )}
                <span className="flex items-center gap-2">
                  {isCurrentlyThinking ? (
                    <>
                      <span className="relative flex h-2 w-2">
                        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-indigo-400 opacity-75" />
                        <span className="relative inline-flex h-2 w-2 rounded-full bg-indigo-400" />
                      </span>
                      <span>Thinking...</span>
                    </>
                  ) : (
                    <span>Reasoning Process</span>
                  )}
                </span>
                {showThinking ? (
                  <ChevronDown size={14} className="ml-auto" />
                ) : (
                  <ChevronRight size={14} className="ml-auto" />
                )}
              </button>
              {showThinking && (
                <div className={`border-t px-4 py-3.5 text-sm leading-relaxed max-h-96 overflow-y-auto ${
                  isCurrentlyThinking 
                    ? 'border-indigo-200 text-slate-600 bg-indigo-50/30' 
                    : 'border-slate-100 text-slate-500'
                }`}>
                  <div className="space-y-2">
                    <ReactMarkdown
                      remarkPlugins={[remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                      className="prose-thinking"
                    >
                      {thinking}
                    </ReactMarkdown>
                    {isCurrentlyThinking && (
                      <div className="flex items-center gap-2 pt-2 text-xs text-indigo-400">
                        <Sparkles size={12} className="animate-pulse" />
                        <span className="italic">Processing thoughts...</span>
                      </div>
                    )}
                  </div>
                  <div ref={thinkingEndRef} />
                </div>
              )}
            </div>
          )}

          {/* Main Output - Clearly Separated */}
          {hasOutput && (
            <>
              {thinking && (
                <div className="flex items-center gap-2 pt-1 pb-1">
                  <div className="h-px flex-1 bg-gradient-to-r from-transparent via-indigo-300 to-transparent" />
                  <span className="text-[10px] uppercase tracking-wider text-indigo-400 flex items-center gap-1">
                    <Zap size={10} />
                    Response
                  </span>
                  <div className="h-px flex-1 bg-gradient-to-r from-transparent via-indigo-300 to-transparent" />
                </div>
              )}
              <div className="prose-chat text-slate-700">
                <ReactMarkdown
                  remarkPlugins={[remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                  components={{
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    code({ className, children, ...props }: any) {
                      const match = /language-(\w+)/.exec(className || "");
                      const isBlock =
                        typeof children === "string" && children.includes("\n");
                      if (isBlock || match) {
                        return (
                          <div className="relative group/code">
                            {match && (
                              <div className="absolute right-3 top-2 text-[10px] text-slate-400 uppercase tracking-wider">
                                {match[1]}
                              </div>
                            )}
                            <pre className="!mt-0">
                              <code className={className} {...props}>
                                {children}
                              </code>
                            </pre>
                          </div>
                        );
                      }
                      return (
                        <code className={className} {...props}>
                          {children}
                        </code>
                      );
                    },
                  }}
                >
                  {displayContent}
                </ReactMarkdown>
              </div>
            </>
          )}

          {/* Streaming cursor */}
          {message.isStreaming && !displayContent && !thinking && (
            <div className="flex items-center gap-1.5 py-2">
              <div className="typing-dot h-1.5 w-1.5 rounded-full bg-slate-400" />
              <div className="typing-dot h-1.5 w-1.5 rounded-full bg-slate-400" />
              <div className="typing-dot h-1.5 w-1.5 rounded-full bg-slate-400" />
            </div>
          )}

          {/* RAG Evidence Cards */}
          {!isUser && message.evidence && message.evidence.length > 0 && (
            <div className="mt-3 space-y-2">
              <div className="flex items-center gap-2">
                <FileText size={12} className="text-slate-400" />
                <span className="text-[11px] font-medium text-slate-500">
                  Evidence ({message.evidence.length} memories)
                </span>
                {message.confidence !== undefined && (
                  <span className={`text-[10px] px-1.5 py-0.5 rounded-md ${
                    message.confidence > 0.7
                      ? "bg-emerald-50 text-emerald-600 border border-emerald-200"
                      : message.confidence > 0.4
                      ? "bg-amber-50 text-amber-600 border border-amber-200"
                      : "bg-red-50 text-red-500 border border-red-200"
                  }`}>
                    {Math.round(message.confidence * 100)}% confidence
                  </span>
                )}
              </div>
              <div className="grid gap-2">
                {message.evidence.slice(0, 3).map((ev, idx) => (
                  <div
                    key={idx}
                    className="rounded-xl border border-slate-200 bg-slate-50/50 p-3 text-xs hover:border-slate-300 transition-colors"
                  >
                    <div className="flex items-center gap-2 mb-1.5">
                      <span className="px-1.5 py-0.5 rounded-md bg-indigo-50 text-indigo-600 text-[9px] font-medium uppercase tracking-wider">
                        {ev.memory_type}
                      </span>
                      <span className="text-slate-400 text-[9px]">
                        {new Date(ev.timestamp).toLocaleDateString()}
                      </span>
                      <span className="text-slate-400 text-[9px]">
                        Score: {ev.score}
                      </span>
                      {ev.channel && (
                        <span className="text-slate-400 text-[9px]">
                          via {ev.channel}
                        </span>
                      )}
                    </div>
                    <p className="text-slate-600 leading-relaxed">
                      {ev.content}
                    </p>
                    {ev.entities && ev.entities.length > 0 && (
                      <div className="flex gap-1 mt-1.5">
                        {ev.entities.map((entity, i) => (
                          <span
                            key={i}
                            className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-500 text-[9px]"
                          >
                            {entity}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* RAG Metadata Bar */}
          {!isUser && !message.isStreaming && (message.agentsUsed || message.queryAnalysis) && (
            <div className="flex flex-wrap items-center gap-2 mt-2 text-[10px] text-slate-400">
              {message.agentsUsed && message.agentsUsed.length > 0 && (
                <span className="flex items-center gap-1 px-2 py-0.5 rounded-lg bg-slate-50 border border-slate-200">
                  <Network size={9} />
                  Agents: {message.agentsUsed.join(", ")}
                </span>
              )}
              {message.queryAnalysis && (
                <span className="flex items-center gap-1 px-2 py-0.5 rounded-lg bg-slate-50 border border-slate-200">
                  <Shield size={9} />
                  {message.queryAnalysis.intent} · {message.queryAnalysis.routing}
                </span>
              )}
              {message.processingTimeMs && (
                <span className="flex items-center gap-1 px-2 py-0.5 rounded-lg bg-slate-50 border border-slate-200">
                  <Timer size={9} />
                  {Math.round(message.processingTimeMs)}ms
                </span>
              )}
              {message.cacheHit && (
                <span className="flex items-center gap-1 px-2 py-0.5 rounded-lg bg-amber-50 border border-amber-200 text-amber-600">
                  <Zap size={9} />
                  Cached
                </span>
              )}
            </div>
          )}

          {/* Footer: usage + actions */}
          {!isUser && !message.isStreaming && displayContent && (
            <div className="flex items-center gap-3 pt-1 opacity-0 group-hover:opacity-100 transition-opacity">
              <button
                onClick={handleCopy}
                className="flex items-center gap-1 text-[10px] text-slate-400 hover:text-slate-600 transition-colors"
              >
                {copied ? <Check size={11} /> : <Copy size={11} />}
                {copied ? "Copied" : "Copy"}
              </button>
              {message.usage && (
                <span className="flex items-center gap-1 text-[10px] text-slate-400">
                  <Zap size={10} />
                  {message.usage.completion_tokens} tokens
                </span>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
