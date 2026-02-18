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
    <div className={`group py-4 ${isUser ? "" : ""}`}>
      <div className="flex gap-3.5">
        {/* Avatar */}
        <div className="flex-shrink-0 mt-0.5">
          {isUser ? (
            <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-deepseek-600/20 text-deepseek-400 ring-1 ring-deepseek-500/20">
              <User size={16} />
            </div>
          ) : (
            <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-emerald-500/10 text-emerald-400 ring-1 ring-emerald-500/20">
              <Bot size={16} />
            </div>
          )}
        </div>

        {/* Content */}
        <div className="min-w-0 flex-1 space-y-2">
          {/* Role label */}
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-surface-300">
              {isUser ? "You" : "Cortex Lab"}
            </span>
            {message.isStreaming && (
              <span className="flex items-center gap-1 text-[10px] text-deepseek-400">
                <span className="relative flex h-1.5 w-1.5">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-deepseek-400 opacity-75" />
                  <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-deepseek-400" />
                </span>
                generating
              </span>
            )}
          </div>

          {/* Live Thinking Panel - Enhanced */}
          {thinking && (
            <div className={`thinking-panel rounded-xl overflow-hidden border transition-all duration-300 ${
              isCurrentlyThinking 
                ? 'border-deepseek-500/30 bg-deepseek-500/5 shadow-lg shadow-deepseek-500/10' 
                : 'border-deepseek-500/10 bg-surface-900/40'
            }`}>
              <button
                onClick={() => setShowThinking((p) => !p)}
                className={`flex w-full items-center gap-2.5 px-4 py-3 text-sm font-medium transition-colors ${
                  isCurrentlyThinking
                    ? 'text-deepseek-300 hover:bg-deepseek-500/10'
                    : 'text-deepseek-400/80 hover:bg-deepseek-500/5'
                }`}
              >
                {isCurrentlyThinking ? (
                  <Brain size={14} className="text-deepseek-400 animate-pulse" />
                ) : (
                  <Brain size={14} className="text-deepseek-400/60" />
                )}
                <span className="flex items-center gap-2">
                  {isCurrentlyThinking ? (
                    <>
                      <span className="relative flex h-2 w-2">
                        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-deepseek-400 opacity-75" />
                        <span className="relative inline-flex h-2 w-2 rounded-full bg-deepseek-400" />
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
                    ? 'border-deepseek-500/20 text-surface-300 bg-deepseek-500/5' 
                    : 'border-deepseek-500/10 text-surface-400'
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
                      <div className="flex items-center gap-2 pt-2 text-xs text-deepseek-400/60">
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
                  <div className="h-px flex-1 bg-gradient-to-r from-transparent via-deepseek-500/20 to-transparent" />
                  <span className="text-[10px] uppercase tracking-wider text-deepseek-400/40 flex items-center gap-1">
                    <Zap size={10} />
                    Response
                  </span>
                  <div className="h-px flex-1 bg-gradient-to-r from-transparent via-deepseek-500/20 to-transparent" />
                </div>
              )}
              <div className="prose-chat text-surface-200">
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
                              <div className="absolute right-3 top-2 text-[10px] text-surface-600 uppercase tracking-wider">
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
              <div className="typing-dot h-1.5 w-1.5 rounded-full bg-surface-500" />
              <div className="typing-dot h-1.5 w-1.5 rounded-full bg-surface-500" />
              <div className="typing-dot h-1.5 w-1.5 rounded-full bg-surface-500" />
            </div>
          )}

          {/* Footer: usage + actions */}
          {!isUser && !message.isStreaming && displayContent && (
            <div className="flex items-center gap-3 pt-1 opacity-0 group-hover:opacity-100 transition-opacity">
              <button
                onClick={handleCopy}
                className="flex items-center gap-1 text-[10px] text-surface-600 hover:text-surface-400 transition-colors"
              >
                {copied ? <Check size={11} /> : <Copy size={11} />}
                {copied ? "Copied" : "Copy"}
              </button>
              {message.usage && (
                <span className="flex items-center gap-1 text-[10px] text-surface-600">
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
