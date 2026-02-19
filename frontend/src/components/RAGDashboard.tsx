"use client";

import { useState, useEffect, useCallback } from "react";
import {
  ArrowLeft,
  Loader2,
  BarChart3,
  Brain,
  Database,
  Network,
  Zap,
  Clock,
  RefreshCw,
  CheckCircle2,
  XCircle,
} from "lucide-react";
import { RAGStats } from "@/lib/types";
import { getRAGStats, getRAGHealth } from "@/lib/api";

export function RAGDashboard({ onBack }: { onBack: () => void }) {
  const [stats, setStats] = useState<RAGStats | null>(null);
  const [health, setHealth] = useState<{ rag_initialized: boolean; stats: RAGStats } | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [s, h] = await Promise.all([getRAGStats(), getRAGHealth()]);
      setStats(s);
      setHealth(h);
    } catch (err) {
      console.error("Failed to load dashboard:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const interval = setInterval(load, 15000);
    return () => clearInterval(interval);
  }, [load]);

  const StatCard = ({
    icon: Icon,
    label,
    value,
    sub,
    color,
  }: {
    icon: any;
    label: string;
    value: string | number;
    sub?: string;
    color: string;
  }) => (
    <div className="rounded-xl border border-surface-800/40 bg-surface-900/30 p-4">
      <div className="flex items-center gap-2 mb-3">
        <div className={`p-1.5 rounded-lg ${color}`}>
          <Icon size={14} />
        </div>
        <span className="text-xs text-surface-500">{label}</span>
      </div>
      <p className="text-2xl font-bold text-surface-100">{value}</p>
      {sub && <p className="text-[10px] text-surface-600 mt-1">{sub}</p>}
    </div>
  );

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
            <BarChart3 size={18} className="text-deepseek-400" />
            <h2 className="text-sm font-semibold text-surface-200">
              RAG Dashboard
            </h2>
          </div>
        </div>
        <button
          onClick={load}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-surface-800/60 text-surface-400 text-xs hover:bg-surface-700/60 transition-colors"
        >
          <RefreshCw size={12} />
          Refresh
        </button>
      </div>

      {loading && !stats ? (
        <div className="flex-1 flex items-center justify-center">
          <Loader2 size={24} className="animate-spin text-deepseek-400" />
        </div>
      ) : !stats ? (
        <div className="flex-1 flex flex-col items-center justify-center text-surface-500">
          <BarChart3 size={40} className="mb-3 opacity-30" />
          <p className="text-sm">RAG system not available</p>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
          {/* Health Status */}
          {health && (
            <div className="rounded-xl border border-surface-800/40 bg-surface-900/30 p-4">
              <div className="flex items-center gap-2 mb-3">
                {health.rag_initialized ? (
                  <CheckCircle2 size={16} className="text-emerald-400" />
                ) : (
                  <XCircle size={16} className="text-red-400" />
                )}
                <h3 className="text-sm font-medium text-surface-200">
                  System Health
                </h3>
                <span
                  className={`ml-auto px-2 py-0.5 rounded-full text-[10px] font-medium ${
                    health.rag_initialized
                      ? "bg-emerald-500/10 text-emerald-400"
                      : "bg-red-500/10 text-red-400"
                  }`}
                >
                  {health.rag_initialized ? "READY" : "NOT INITIALIZED"}
                </span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                {[
                  { name: "RAG Engine", ok: health.rag_initialized },
                  { name: "Vector Store", ok: !!health.stats?.vectors },
                  { name: "Metadata Store", ok: !!health.stats?.memories },
                  { name: "Knowledge Graph", ok: !!health.stats?.graph },
                  { name: "Cache", ok: !!health.stats?.cache },
                  { name: "LLM", ok: health.stats?.llm?.model_loaded ?? false },
                ].map((comp) => (
                  <div
                    key={comp.name}
                    className="flex items-center gap-2 px-2 py-1.5 rounded-lg bg-surface-800/30"
                  >
                    <div
                      className={`w-1.5 h-1.5 rounded-full ${
                        comp.ok ? "bg-emerald-400" : "bg-red-400"
                      }`}
                    />
                    <span className="text-[10px] text-surface-400">
                      {comp.name}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Stats Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
            <StatCard
              icon={Brain}
              label="Total Memories"
              value={stats.memories?.memories ?? 0}
              color="bg-deepseek-500/10 text-deepseek-400"
            />
            <StatCard
              icon={Network}
              label="Entities"
              value={stats.memories?.entities ?? 0}
              color="bg-purple-500/10 text-purple-400"
            />
            <StatCard
              icon={Database}
              label="Graph Edges"
              value={stats.memories?.edges ?? 0}
              color="bg-blue-500/10 text-blue-400"
            />
            <StatCard
              icon={Zap}
              label="Belief Deltas"
              value={stats.memories?.belief_deltas ?? 0}
              color="bg-amber-500/10 text-amber-400"
            />
          </div>

          {/* Vectors */}
          {stats.vectors && (
            <div className="rounded-xl border border-surface-800/40 bg-surface-900/30 p-4">
              <h3 className="text-xs font-medium text-surface-400 mb-3">
                Vector Store
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div>
                  <p className="text-lg font-bold text-surface-100">
                    {stats.vectors.total_vectors ?? 0}
                  </p>
                  <p className="text-[10px] text-surface-600">Total Vectors</p>
                </div>
                <div>
                  <p className="text-lg font-bold text-surface-100">
                    {stats.vectors.dimension ?? 0}
                  </p>
                  <p className="text-[10px] text-surface-600">Dimensions</p>
                </div>
                <div>
                  <p className="text-lg font-bold text-surface-100">
                    {stats.vectors.hot_count ?? 0}
                  </p>
                  <p className="text-[10px] text-surface-600">Hot Vectors</p>
                </div>
                <div>
                  <p className="text-lg font-bold text-surface-100">
                    {stats.vectors.cold_count ?? 0}
                  </p>
                  <p className="text-[10px] text-surface-600">Cold Vectors</p>
                </div>
              </div>
            </div>
          )}

          {/* Graph Stats */}
          {stats.graph && (
            <div className="rounded-xl border border-surface-800/40 bg-surface-900/30 p-4">
              <h3 className="text-xs font-medium text-surface-400 mb-3">
                Knowledge Graph
              </h3>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <p className="text-lg font-bold text-surface-100">
                    {stats.graph.nodes ?? 0}
                  </p>
                  <p className="text-[10px] text-surface-600">Nodes</p>
                </div>
                <div>
                  <p className="text-lg font-bold text-surface-100">
                    {stats.graph.edges ?? 0}
                  </p>
                  <p className="text-[10px] text-surface-600">Edges</p>
                </div>
              </div>
            </div>
          )}

          {/* Cache */}
          {stats.cache && (
            <div className="rounded-xl border border-surface-800/40 bg-surface-900/30 p-4">
              <h3 className="text-xs font-medium text-surface-400 mb-3">
                Cache Performance
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div>
                  <p className="text-lg font-bold text-emerald-400">
                    {((stats.cache.hit_rate ?? 0) * 100).toFixed(1)}%
                  </p>
                  <p className="text-[10px] text-surface-600">Hit Rate</p>
                </div>
                <div>
                  <p className="text-lg font-bold text-surface-100">
                    {stats.cache.total_hits ?? 0}
                  </p>
                  <p className="text-[10px] text-surface-600">Hits</p>
                </div>
                <div>
                  <p className="text-lg font-bold text-surface-100">
                    {stats.cache.total_queries ?? 0}
                  </p>
                  <p className="text-[10px] text-surface-600">Total Queries</p>
                </div>
                <div>
                  <p className="text-lg font-bold text-surface-100">
                    {(stats.cache.exact_cache_size ?? 0) + (stats.cache.semantic_cache_size ?? 0)}
                  </p>
                  <p className="text-[10px] text-surface-600">Cached Entries</p>
                </div>
              </div>
            </div>
          )}

          {/* LLM Stats */}
          {stats.llm && (
            <div className="rounded-xl border border-surface-800/40 bg-surface-900/30 p-4">
              <h3 className="text-xs font-medium text-surface-400 mb-3">
                LLM Usage
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                <div>
                  <p className="text-lg font-bold text-surface-100">
                    {stats.llm.call_count ?? 0}
                  </p>
                  <p className="text-[10px] text-surface-600">Total Calls</p>
                </div>
                <div>
                  <p className="text-lg font-bold text-surface-100">
                    {stats.llm.total_tokens ?? 0}
                  </p>
                  <p className="text-[10px] text-surface-600">Total Tokens</p>
                </div>
                <div>
                  <p className="text-lg font-bold text-surface-100">
                    {stats.llm.model_loaded ? "✓ Loaded" : "✗ Not loaded"}
                  </p>
                  <p className="text-[10px] text-surface-600">Model Status</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
