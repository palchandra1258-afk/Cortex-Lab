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
    happy: "text-emerald-600 bg-emerald-50",
    sad: "text-blue-600 bg-blue-50",
    angry: "text-red-600 bg-red-50",
    anxious: "text-amber-600 bg-amber-50",
    neutral: "text-slate-500 bg-slate-100",
    excited: "text-purple-600 bg-purple-50",
    confused: "text-orange-600 bg-orange-50",
    hopeful: "text-cyan-600 bg-cyan-50",
    frustrated: "text-red-500 bg-red-50",
  };

  const typeColors: Record<string, string> = {
    episodic: "bg-indigo-50 text-indigo-600 border border-indigo-200",
    semantic: "bg-purple-50 text-purple-600 border border-purple-200",
    procedural: "bg-amber-50 text-amber-600 border border-amber-200",
    reflective: "bg-emerald-50 text-emerald-600 border border-emerald-200",
  };

  return (
    <div className="flex flex-1 flex-col overflow-hidden bg-white">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-200 px-5 py-3.5">
        <div className="flex items-center gap-3">
          <button
            onClick={onBack}
            className="rounded-lg p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-all duration-200"
          >
            <ArrowLeft size={18} />
          </button>
          <div className="flex items-center gap-2.5">
            <Brain size={18} className="text-indigo-500" />
            <h2 className="text-sm font-semibold text-slate-700">
              Memory Browser
            </h2>
            <span className="text-[10px] text-slate-500 bg-slate-100 border border-slate-200 px-2 py-0.5 rounded-lg">
              {total} memories
            </span>
          </div>
        </div>
        <button
          onClick={() => setShowIngest(!showIngest)}
          className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-gradient-to-r from-indigo-600 to-indigo-500 text-white text-xs font-medium hover:from-indigo-500 hover:to-indigo-400 transition-all duration-200 shadow-lg shadow-indigo-200/50"
        >
          <Plus size={14} />
          Add Memory
        </button>
      </div>

      {/* Search + Ingest */}
      <div className="px-5 py-3.5 space-y-3 border-b border-slate-200">
        <div className="flex gap-2">
          <div className="flex-1 relative">
            <Search
              size={14}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
            />
            <input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="Search memories semantically..."
              className="w-full rounded-xl bg-slate-50 border border-slate-200 pl-9 pr-3 py-2.5 text-sm text-slate-800 placeholder:text-slate-400 outline-none focus:border-indigo-300 focus:bg-white transition-all duration-200"
            />
          </div>
          <button
            onClick={handleSearch}
            disabled={isSearching}
            className="px-4 py-2.5 rounded-xl bg-slate-50 border border-slate-200 text-slate-600 text-sm hover:bg-slate-100 hover:text-slate-800 transition-all duration-200 disabled:opacity-50"
          >
            {isSearching ? <Loader2 size={14} className="animate-spin" /> : "Search"}
          </button>
        </div>

        {showIngest && (
          <div className="flex gap-2 fade-in">
            <textarea
              value={newMemory}
              onChange={(e) => setNewMemory(e.target.value)}
              placeholder="Enter a memory to store... (e.g., 'Had a great meeting with Sarah about Project X today')"
              rows={2}
              className="flex-1 rounded-xl bg-slate-50 border border-slate-200 px-3.5 py-2.5 text-sm text-slate-800 placeholder:text-slate-400 outline-none focus:border-indigo-300 transition-all duration-200 resize-none"
            />
            <button
              onClick={handleIngest}
              disabled={ingesting || !newMemory.trim()}
              className="px-4 rounded-xl bg-emerald-600 text-white text-sm font-medium hover:bg-emerald-500 transition-all duration-200 disabled:opacity-50 shadow-lg shadow-emerald-200/50"
            >
              {ingesting ? <Loader2 size={14} className="animate-spin" /> : "Store"}
            </button>
          </div>
        )}
      </div>

      {/* Memory List */}
      <div className="flex-1 overflow-y-auto px-5 py-3 space-y-2.5">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 size={24} className="animate-spin text-indigo-500" />
          </div>
        ) : memories.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-slate-400 fade-in">
            <Brain size={40} className="mb-3 opacity-30" />
            <p className="text-sm font-medium">No memories yet</p>
            <p className="text-xs mt-1 text-slate-400">Chat with Cortex Lab or add memories manually</p>
          </div>
        ) : (
          memories.map((mem) => (
            <div
              key={mem.id}
              className="group rounded-2xl border border-slate-200 bg-white p-4 hover:border-slate-300 hover:bg-slate-50/50 transition-all duration-200 card-hover"
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
                <span className="flex items-center gap-1 text-[10px] text-slate-400">
                  <Clock size={9} />
                  {new Date(mem.timestamp).toLocaleString()}
                </span>
                <span className="text-[10px] text-slate-400 ml-auto">
                  importance: {(mem.importance * 100).toFixed(0)}%
                </span>
                {mem.score !== undefined && (
                  <span className="text-[10px] text-indigo-500">
                    match: {(mem.score * 100).toFixed(0)}%
                  </span>
                )}
              </div>

              {/* Content */}
              <p className="text-sm text-slate-700 leading-relaxed mb-2">
                {mem.content}
              </p>

              {/* Tags row: topics + entities */}
              <div className="flex flex-wrap gap-1.5">
                {mem.topics?.map((topic) => (
                  <span
                    key={topic}
                    className="flex items-center gap-1 px-1.5 py-0.5 rounded-md bg-indigo-50 text-indigo-600 text-[9px] border border-indigo-200"
                  >
                    <Tag size={8} />
                    {topic}
                  </span>
                ))}
                {mem.entities?.map((entity) => (
                  <span
                    key={entity}
                    className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-500 text-[9px]"
                  >
                    {entity}
                  </span>
                ))}
              </div>

              {/* Delete button */}
              <div className="flex justify-end mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                  onClick={() => handleDelete(mem.id)}
                  className="flex items-center gap-1 text-[10px] text-red-400 hover:text-red-500 transition-colors"
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
