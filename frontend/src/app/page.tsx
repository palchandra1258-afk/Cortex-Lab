"use client";

import { useState, useEffect } from "react";
import { ChatPanel } from "@/components/ChatPanel";
import { Sidebar } from "@/components/Sidebar";
import { Header } from "@/components/Header";
import { ModelStatus } from "@/lib/types";

export default function Home() {
  const [modelStatus, setModelStatus] = useState<ModelStatus>({
    status: "loading",
    model_loaded: false,
    model_info: {},
  });
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [conversations, setConversations] = useState<
    { id: string; title: string; date: string }[]
  >([{ id: "1", title: "New Chat", date: new Date().toISOString() }]);
  const [activeConversation, setActiveConversation] = useState("1");

  // Poll model health
  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch("/api/health");
        if (res.ok) {
          const data = await res.json();
          setModelStatus(data);
        }
      } catch {
        setModelStatus({
          status: "offline",
          model_loaded: false,
          model_info: {},
        });
      }
    };
    check();
    const interval = setInterval(check, 10_000);
    return () => clearInterval(interval);
  }, []);

  const handleNewChat = () => {
    const id = Date.now().toString();
    setConversations((prev) => [
      { id, title: "New Chat", date: new Date().toISOString() },
      ...prev,
    ]);
    setActiveConversation(id);
  };

  return (
    <div className="flex h-screen overflow-hidden bg-surface-950">
      {/* Sidebar */}
      <Sidebar
        open={sidebarOpen}
        conversations={conversations}
        activeId={activeConversation}
        onSelect={setActiveConversation}
        onNewChat={handleNewChat}
        onToggle={() => setSidebarOpen((p) => !p)}
      />

      {/* Main Area */}
      <div className="flex flex-1 flex-col min-w-0">
        <Header
          modelStatus={modelStatus}
          sidebarOpen={sidebarOpen}
          onToggleSidebar={() => setSidebarOpen((p) => !p)}
        />
        <ChatPanel
          key={activeConversation}
          modelStatus={modelStatus}
          onTitleUpdate={(title) =>
            setConversations((prev) =>
              prev.map((c) =>
                c.id === activeConversation ? { ...c, title } : c,
              ),
            )
          }
        />
      </div>
    </div>
  );
}
