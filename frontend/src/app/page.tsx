"use client";

import { useState, useEffect } from "react";
import { ChatPanel } from "@/components/ChatPanel";
import { Sidebar } from "@/components/Sidebar";
import { Header } from "@/components/Header";
import { MemoryBrowser } from "@/components/MemoryBrowser";
import { KnowledgeGraph } from "@/components/KnowledgeGraph";
import { RAGDashboard } from "@/components/RAGDashboard";
import { ModelStatus } from "@/lib/types";

type ActiveView = "chat" | "memories" | "graph" | "dashboard";

export default function Home() {
  const [modelStatus, setModelStatus] = useState<ModelStatus>({
    status: "loading",
    model_loaded: false,
    model_info: {},
  });
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeView, setActiveView] = useState<ActiveView>("chat");
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
    setActiveView("chat");
  };

  return (
    <div className="flex h-screen overflow-hidden bg-[#f8fafc]">
      {/* Ambient background glow */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-[40%] -left-[20%] w-[70%] h-[70%] rounded-full bg-indigo-500/[0.03] blur-[120px]" />
        <div className="absolute -bottom-[30%] -right-[20%] w-[60%] h-[60%] rounded-full bg-violet-500/[0.03] blur-[120px]" />
      </div>

      {/* Sidebar */}
      <Sidebar
        open={sidebarOpen}
        conversations={conversations}
        activeId={activeConversation}
        onSelect={(id) => {
          setActiveConversation(id);
          setActiveView("chat");
        }}
        onNewChat={handleNewChat}
        onToggle={() => setSidebarOpen((p) => !p)}
        activeView={activeView}
        onNavigate={setActiveView}
      />

      {/* Main Area */}
      <div className="flex flex-1 flex-col min-w-0">
        <Header
          modelStatus={modelStatus}
          sidebarOpen={sidebarOpen}
          onToggleSidebar={() => setSidebarOpen((p) => !p)}
        />
        {activeView === "chat" && (
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
        )}
        {activeView === "memories" && (
          <MemoryBrowser onBack={() => setActiveView("chat")} />
        )}
        {activeView === "graph" && (
          <KnowledgeGraph onBack={() => setActiveView("chat")} />
        )}
        {activeView === "dashboard" && (
          <RAGDashboard onBack={() => setActiveView("chat")} />
        )}
      </div>
    </div>
  );
}
