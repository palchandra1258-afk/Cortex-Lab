"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Send, Square, Settings2, Sparkles, Brain } from "lucide-react";
import { ChatMessage as ChatMessageType, ModelStatus, ChatSettings, DEFAULT_SETTINGS } from "@/lib/types";
import { sendMessage, streamMessage, ragChat, streamRAGMessage, RAGStreamMeta } from "@/lib/api";
import { MessageBubble } from "./MessageBubble";
import { SettingsPanel } from "./SettingsPanel";
import { EmptyState } from "./EmptyState";

interface Props {
  modelStatus: ModelStatus;
  onTitleUpdate: (title: string) => void;
}

export function ChatPanel({ modelStatus, onTitleUpdate }: Props) {
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [input, setInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState<ChatSettings>(DEFAULT_SETTINGS);
  const [error, setError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef(false);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-resize textarea
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  };

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || isGenerating) return;

    setError(null);
    abortRef.current = false;

    // Add user message
    const userMsg: ChatMessageType = {
      id: `user-${Date.now()}`,
      role: "user",
      content: text,
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsGenerating(true);

    // Reset textarea height
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
    }

    // Update conversation title from first message
    if (messages.length === 0) {
      onTitleUpdate(text.slice(0, 50) + (text.length > 50 ? "…" : ""));
    }

    // Prepare history
    const history = [...messages, userMsg].map((m) => ({
      role: m.role,
      content: m.content,
    }));

    // Create assistant placeholder
    const assistantId = `assistant-${Date.now()}`;

    // ── RAG-Enhanced Mode ───────────────────────────────────────
    if (settings.useRAG) {
      if (settings.stream) {
        // ── RAG + Streaming ─────────────────────────────────────
        const assistantMsg: ChatMessageType = {
          id: assistantId,
          role: "assistant",
          content: "",
          timestamp: Date.now(),
          isStreaming: true,
        };
        setMessages((prev) => [...prev, assistantMsg]);

        await streamRAGMessage(
          history,
          settings,
          "", // session_id
          (meta: RAGStreamMeta) => {
            // Update message with RAG metadata (evidence, agents, etc.)
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      thinking: meta.thinking || m.thinking,
                      evidence: meta.evidence || m.evidence,
                      agentsUsed: meta.agents_used || m.agentsUsed,
                      confidence: meta.confidence ?? m.confidence,
                      queryAnalysis: meta.query_analysis || m.queryAnalysis,
                    }
                  : m,
              ),
            );
          },
          (token) => {
            if (abortRef.current) return;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: m.content + token }
                  : m,
              ),
            );
          },
          () => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, isStreaming: false } : m,
              ),
            );
            setIsGenerating(false);
          },
          (err) => {
            setError(err.message);
            setIsGenerating(false);
            setMessages((prev) =>
              prev.filter((m) => m.id !== assistantId || m.content.length > 0),
            );
          },
        );
      } else {
        // ── RAG Non-streaming ───────────────────────────────────
        try {
          const res = await ragChat(history, settings);
          const assistantMsg: ChatMessageType = {
            id: assistantId,
            role: "assistant",
            content: res.content,
            thinking: res.thinking || undefined,
            timestamp: Date.now(),
            evidence: res.evidence,
            agentsUsed: res.agents_used,
            confidence: res.confidence,
            queryAnalysis: res.query_analysis,
            processingTimeMs: res.processing_time_ms,
            cacheHit: res.cache_hit,
          };
          setMessages((prev) => [...prev, assistantMsg]);
        } catch (err: unknown) {
          const message = err instanceof Error ? err.message : "Unknown error";
          setError(message);
        } finally {
          setIsGenerating(false);
        }
      }
      return;
    }

    if (settings.stream) {
      // ── Streaming ─────────────────────────────────────────────
      const assistantMsg: ChatMessageType = {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: Date.now(),
        isStreaming: true,
      };
      setMessages((prev) => [...prev, assistantMsg]);

      await streamMessage(
        history,
        settings,
        (token) => {
          if (abortRef.current) return;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, content: m.content + token }
                : m,
            ),
          );
        },
        () => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, isStreaming: false } : m,
            ),
          );
          setIsGenerating(false);
        },
        (err) => {
          setError(err.message);
          setIsGenerating(false);
          // Remove empty assistant message on error
          setMessages((prev) =>
            prev.filter((m) => m.id !== assistantId || m.content.length > 0),
          );
        },
      );
    } else {
      // ── Non-streaming ─────────────────────────────────────────
      try {
        const res = await sendMessage(history, settings);
        const assistantMsg: ChatMessageType = {
          id: assistantId,
          role: "assistant",
          content: res.content,
          thinking: res.thinking || undefined,
          timestamp: Date.now(),
          usage: res.usage,
        };
        setMessages((prev) => [...prev, assistantMsg]);
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : "Unknown error";
        setError(message);
      } finally {
        setIsGenerating(false);
      }
    }
  }, [input, isGenerating, messages, settings, onTitleUpdate]);

  const handleStop = () => {
    abortRef.current = true;
    setIsGenerating(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const isOnline = modelStatus.model_loaded;

  return (
    <div className="flex flex-1 flex-col overflow-hidden relative">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="mx-auto max-w-3xl space-y-1">
          {messages.length === 0 ? (
            <EmptyState
              isOnline={isOnline}
              onSuggestion={(text: string) => {
                setInput(text);
                inputRef.current?.focus();
              }}
            />
          ) : (
            messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Error toast */}
      {error && (
        <div className="absolute bottom-28 left-1/2 -translate-x-1/2 z-50 fade-in">
          <div className="rounded-xl bg-red-500/[0.07] border border-red-500/20 px-4 py-2.5 text-sm text-red-400 backdrop-blur-xl flex items-center gap-2 shadow-2xl shadow-red-500/10">
            <span>⚠️ {error}</span>
            <button
              onClick={() => setError(null)}
              className="text-red-400/60 hover:text-red-300 ml-2 transition-colors"
            >
              ✕
            </button>
          </div>
        </div>
      )}

      {/* Settings panel */}
      {showSettings && (
        <SettingsPanel
          settings={settings}
          onUpdate={setSettings}
          onClose={() => setShowSettings(false)}
        />
      )}

      {/* Input area */}
      <div className="border-t border-slate-200/80 bg-white/90 backdrop-blur-2xl px-4 py-4">
        <div className="mx-auto max-w-3xl">
          <div className="glow-border rounded-2xl bg-white transition-all duration-300 border border-slate-200">
            <div className="flex items-end gap-2 p-3">
              {/* Settings button */}
              <button
                onClick={() => setShowSettings((p) => !p)}
                className={`flex-shrink-0 rounded-xl p-2.5 transition-all duration-200 ${
                  showSettings
                    ? "bg-indigo-50 text-indigo-600 shadow-sm shadow-indigo-100"
                    : "text-slate-400 hover:text-slate-600 hover:bg-slate-100"
                }`}
                title="Settings"
              >
                <Settings2 size={18} />
              </button>

              {/* Input */}
              <textarea
                ref={inputRef}
                value={input}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                placeholder={
                  isOnline
                    ? "Ask Cortex Lab anything…"
                    : "Model is loading — please wait…"
                }
                disabled={!isOnline}
                rows={1}
                className="flex-1 resize-none bg-transparent text-sm text-slate-800 placeholder:text-slate-400 outline-none disabled:opacity-40 py-2.5 leading-relaxed"
              />

              {/* Send / Stop */}
              {isGenerating ? (
                <button
                  onClick={handleStop}
                  className="flex-shrink-0 rounded-xl bg-red-50 p-2.5 text-red-500 hover:bg-red-100 border border-red-200 transition-all duration-200"
                  title="Stop generating"
                >
                  <Square size={18} />
                </button>
              ) : (
                <button
                  onClick={handleSend}
                  disabled={!input.trim() || !isOnline}
                  className="flex-shrink-0 rounded-xl bg-gradient-to-r from-indigo-600 to-indigo-500 p-2.5 text-white transition-all duration-200 hover:from-indigo-500 hover:to-indigo-400 disabled:opacity-20 disabled:hover:from-indigo-600 disabled:hover:to-indigo-500 shadow-lg shadow-indigo-200/50"
                  title="Send message"
                >
                  <Send size={18} />
                </button>
              )}
            </div>

            {/* Bottom bar */}
            <div className="flex items-center justify-between border-t border-slate-100 px-4 py-2 text-[11px] text-slate-400">
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5">
                  <Sparkles size={11} className="text-indigo-500/60" />
                  <span>DeepSeek-R1-7B Fine-Tuned</span>
                </div>
                <button
                  onClick={() => setSettings((prev) => ({ ...prev, useRAG: !prev.useRAG }))}
                  className={`flex items-center gap-1 px-2 py-0.5 rounded-md transition-all text-[10px] font-medium ${
                    settings.useRAG
                      ? "bg-emerald-50 text-emerald-600 border border-emerald-200"
                      : "bg-slate-50 text-slate-400 border border-slate-200"
                  }`}
                >
                  <Brain size={10} />
                  {settings.useRAG ? "RAG ON" : "RAG OFF"}
                </button>
              </div>
              <span className="text-slate-400">
                Temp {settings.temperature} · Top-P {settings.topP} · Max{" "}
                {settings.maxTokens}
              </span>
            </div>
          </div>

          <p className="mt-2.5 text-center text-[10px] text-slate-400">
            Shift+Enter for new line · Enter to send · Model may produce inaccurate responses
          </p>
        </div>
      </div>
    </div>
  );
}
