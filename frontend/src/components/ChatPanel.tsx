"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Send, Square, Settings2, Sparkles } from "lucide-react";
import { ChatMessage as ChatMessageType, ModelStatus, ChatSettings, DEFAULT_SETTINGS } from "@/lib/types";
import { sendMessage, streamMessage } from "@/lib/api";
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
        <div className="absolute bottom-28 left-1/2 -translate-x-1/2 z-50">
          <div className="rounded-lg bg-red-500/10 border border-red-500/30 px-4 py-2.5 text-sm text-red-400 backdrop-blur-sm flex items-center gap-2 shadow-xl">
            <span>⚠️ {error}</span>
            <button
              onClick={() => setError(null)}
              className="text-red-400/60 hover:text-red-300 ml-2"
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
      <div className="border-t border-surface-800/50 bg-surface-950/80 backdrop-blur-xl px-4 py-4">
        <div className="mx-auto max-w-3xl">
          <div className="glow-border rounded-2xl bg-surface-900/60 transition-all duration-300">
            <div className="flex items-end gap-2 p-3">
              {/* Settings button */}
              <button
                onClick={() => setShowSettings((p) => !p)}
                className={`flex-shrink-0 rounded-xl p-2.5 transition-all duration-200 ${
                  showSettings
                    ? "bg-deepseek-600/20 text-deepseek-400"
                    : "text-surface-500 hover:text-surface-300 hover:bg-surface-800/60"
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
                    ? "Ask DeepSeek anything…"
                    : "Model is loading — please wait…"
                }
                disabled={!isOnline}
                rows={1}
                className="flex-1 resize-none bg-transparent text-sm text-surface-100 placeholder:text-surface-600 outline-none disabled:opacity-40 py-2.5 leading-relaxed"
              />

              {/* Send / Stop */}
              {isGenerating ? (
                <button
                  onClick={handleStop}
                  className="flex-shrink-0 rounded-xl bg-red-500/15 p-2.5 text-red-400 hover:bg-red-500/25 transition-all duration-200"
                  title="Stop generating"
                >
                  <Square size={18} />
                </button>
              ) : (
                <button
                  onClick={handleSend}
                  disabled={!input.trim() || !isOnline}
                  className="flex-shrink-0 rounded-xl bg-deepseek-600 p-2.5 text-white transition-all duration-200 hover:bg-deepseek-500 disabled:opacity-30 disabled:hover:bg-deepseek-600 shadow-lg shadow-deepseek-600/20"
                  title="Send message"
                >
                  <Send size={18} />
                </button>
              )}
            </div>

            {/* Bottom bar */}
            <div className="flex items-center justify-between border-t border-surface-800/30 px-4 py-2 text-[11px] text-surface-600">
              <div className="flex items-center gap-1.5">
                <Sparkles size={11} />
                <span>DeepSeek-R1-Distill-Qwen-14B</span>
              </div>
              <span>
                Temp {settings.temperature} · Top-P {settings.topP} · Max{" "}
                {settings.maxTokens}
              </span>
            </div>
          </div>

          <p className="mt-2 text-center text-[10px] text-surface-700">
            Shift+Enter for new line · Enter to send · Model may produce inaccurate responses
          </p>
        </div>
      </div>
    </div>
  );
}
