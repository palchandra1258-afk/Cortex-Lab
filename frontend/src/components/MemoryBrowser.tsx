"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Brain,
  Search,
  Plus,
  Trash2,
  Clock,
  Tag,
  Heart,
  FileText,
  ArrowLeft,
  Loader2,
  ChevronDown,
} from "lucide-react";
import { MemoryObject } from "@/lib/types";
import { getMemories, searchMemories, ingestMemory, deleteMemory } from "@/lib/api";

export function MemoryBrowser({ onBack }: { onBack: () => void }) {
  const [memories, setMemories] = useState<MemoryObject[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [showIngest, setShowIngest] = useState(false);
  const [newMemory, setNewMemory] = useState("");
  const [ingesting, setIngesting] = useState(false);

  const loadMemories = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getMemories(50, 0);
      setMemories(data.memories);
      setTotal(data.total);
    } catch (err) {
      console.error("Failed to load memories:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadMemories();
  }, [loadMemories]);

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      loadMemories();
      return;
    }
    setIsSearching(true);
    try {
      const data = await searchMemories(searchQuery);
      setMemories(data.results);
      setTotal(data.count);
    } catch (err) {
      console.error("Search failed:", err);
    } finally {
      setIsSearching(false);
    }
  };

  const handleIngest = async () => {
    if (!newMemory.trim()) return;
    setIngesting(true);
    try {
      await ingestMemory(newMemory);
      setNewMemory("");
      setShowIngest(false);
      loadMemories();
    } catch (err) {
      console.error("Ingest failed:", err);
    } finally {
      setIngesting(false);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteMemory(id);
      setMemories((prev) => prev.filter((m) => m.id !== id));
      setTotal((prev) => prev - 1);
    } catch (err) {
      console.error("Delete failed:", err);
    }
  };

  const emotionColors: Record<string, string> = {
    happy: "text-emerald-400 bg-emerald-500/10",
    sad: "text-blue-400 bg-blue-500/10",
    angry: "text-red-400 bg-red-500/10",
    anxious: "text-amber-400 bg-amber-500/10",
    neutral: "text-surface-400 bg-surface-800/50",
    excited: "text-purple-400 bg-purple-500/10",
    confused: "text-orange-400 bg-orange-500/10",
    hopeful: "text-cyan-400 bg-cyan-500/10",
    frustrated: "text-red-300 bg-red-500/10",
  };

  const typeColors: Record<string, string> = {
    episodic: "bg-deepseek-500/10 text-deepseek-400",
    semantic: "bg-purple-500/10 text-purple-400",
    procedural: "bg-amber-500/10 text-amber-400",
    reflective: "bg-emerald-500/10 text-emerald-400",
  };

  return (
    <div className="flex flex-1 flex-col overflow-hidden bg-surface-950">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-surface-800/50 px-4 py-3">
        <div className="flex items-center gap-3">
          <button
            onClick={onBack}
            className="rounded-lg p-1.5 text-surface-500 hover:text-surface-300 hover:bg-surface-800/50 transition-colors"
          >
            <ArrowLeft size={18} />
          </button>
          <div className="flex items-center gap-2">
            <Brain size={18} className="text-deepseek-400" />
            <h2 className="text-sm font-semibold text-surface-200">
              Memory Browser
            </h2>
            <span className="text-[10px] text-surface-500 bg-surface-800/60 px-2 py-0.5 rounded-md">
              {total} memories
            </span>
          </div>
        </div>
        <button
          onClick={() => setShowIngest(!showIngest)}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-deepseek-600 text-white text-xs font-medium hover:bg-deepseek-500 transition-colors"
        >
          <Plus size={14} />
          Add Memory
        </button>
      </div>

      {/* Search + Ingest */}
      <div className="px-4 py-3 space-y-3 border-b border-surface-800/30">
        <div className="flex gap-2">
          <div className="flex-1 relative">
            <Search
              size={14}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-surface-500"
            />
            <input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="Search memories semantically..."
              className="w-full rounded-lg bg-surface-900/60 border border-surface-800/50 pl-9 pr-3 py-2 text-sm text-surface-200 placeholder:text-surface-600 outline-none focus:border-deepseek-500/40"
            />
          </div>
          <button
            onClick={handleSearch}
            disabled={isSearching}
            className="px-4 py-2 rounded-lg bg-surface-800/60 text-surface-300 text-sm hover:bg-surface-700/60 transition-colors disabled:opacity-50"
          >
            {isSearching ? <Loader2 size={14} className="animate-spin" /> : "Search"}
          </button>
        </div>

        {showIngest && (
          <div className="flex gap-2">
            <textarea
              value={newMemory}
              onChange={(e) => setNewMemory(e.target.value)}
              placeholder="Enter a memory to store... (e.g., 'Had a great meeting with Sarah about Project X today')"
              rows={2}
              className="flex-1 rounded-lg bg-surface-900/60 border border-surface-800/50 px-3 py-2 text-sm text-surface-200 placeholder:text-surface-600 outline-none focus:border-deepseek-500/40 resize-none"
            />
            <button
              onClick={handleIngest}
              disabled={ingesting || !newMemory.trim()}
              className="px-4 rounded-lg bg-emerald-600 text-white text-sm font-medium hover:bg-emerald-500 transition-colors disabled:opacity-50"
            >
              {ingesting ? <Loader2 size={14} className="animate-spin" /> : "Store"}
            </button>
          </div>
        )}
      </div>

      {/* Memory List */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-2">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 size={24} className="animate-spin text-deepseek-400" />
          </div>
        ) : memories.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-surface-500">
            <Brain size={40} className="mb-3 opacity-30" />
            <p className="text-sm">No memories yet</p>
            <p className="text-xs mt-1">Chat with Cortex Lab or add memories manually</p>
          </div>
        ) : (
          memories.map((mem) => (
            <div
              key={mem.id}
              className="group rounded-xl border border-surface-800/40 bg-surface-900/30 p-4 hover:border-surface-700/50 transition-all"
            >
              {/* Top row: type + emotion + time */}
              <div className="flex items-center gap-2 mb-2">
                <span
                  className={`px-1.5 py-0.5 rounded text-[9px] font-medium uppercase ${
                    typeColors[mem.memory_type] || typeColors.episodic
                  }`}
                >
                  {mem.memory_type}
                </span>
                <span
                  className={`px-1.5 py-0.5 rounded text-[9px] ${
                    emotionColors[mem.emotion] || emotionColors.neutral
                  }`}
                >
                  {mem.emotion}
                </span>
                <span className="flex items-center gap-1 text-[10px] text-surface-600">
                  <Clock size={9} />
                  {new Date(mem.timestamp).toLocaleString()}
                </span>
                <span className="text-[10px] text-surface-600 ml-auto">
                  importance: {(mem.importance * 100).toFixed(0)}%
                </span>
                {mem.score !== undefined && (
                  <span className="text-[10px] text-deepseek-400">
                    match: {(mem.score * 100).toFixed(0)}%
                  </span>
                )}
              </div>

              {/* Content */}
              <p className="text-sm text-surface-200 leading-relaxed mb-2">
                {mem.content}
              </p>

              {/* Tags row: topics + entities */}
              <div className="flex flex-wrap gap-1.5">
                {mem.topics?.map((topic) => (
                  <span
                    key={topic}
                    className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-deepseek-500/10 text-deepseek-400 text-[9px]"
                  >
                    <Tag size={8} />
                    {topic}
                  </span>
                ))}
                {mem.entities?.map((entity) => (
                  <span
                    key={entity}
                    className="px-1.5 py-0.5 rounded bg-surface-800/60 text-surface-400 text-[9px]"
                  >
                    {entity}
                  </span>
                ))}
              </div>

              {/* Delete button */}
              <div className="flex justify-end mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                  onClick={() => handleDelete(mem.id)}
                  className="flex items-center gap-1 text-[10px] text-red-400/60 hover:text-red-400 transition-colors"
                >
                  <Trash2 size={10} />
                  Delete
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
