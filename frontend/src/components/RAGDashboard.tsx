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
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    icon: any;
    label: string;
    value: string | number;
    sub?: string;
    color: string;
  }) => (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 card-hover">
      <div className="flex items-center gap-2 mb-3">
        <div className={`p-1.5 rounded-xl ${color}`}>
          <Icon size={14} />
        </div>
        <span className="text-xs text-slate-500">{label}</span>
      </div>
      <p className="text-2xl font-bold text-slate-800">{value}</p>
      {sub && <p className="text-[10px] text-slate-400 mt-1">{sub}</p>}
    </div>
  );

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
            <BarChart3 size={18} className="text-indigo-500" />
            <h2 className="text-sm font-semibold text-slate-700">
              RAG Dashboard
            </h2>
          </div>
        </div>
        <button
          onClick={load}
          className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-slate-50 border border-slate-200 text-slate-500 text-xs hover:bg-slate-100 hover:text-slate-700 transition-all duration-200"
        >
          <RefreshCw size={12} />
          Refresh
        </button>
      </div>

      {loading && !stats ? (
        <div className="flex-1 flex items-center justify-center">
          <Loader2 size={24} className="animate-spin text-indigo-500" />
        </div>
      ) : !stats ? (
        <div className="flex-1 flex flex-col items-center justify-center text-slate-400 fade-in">
          <BarChart3 size={40} className="mb-3 opacity-30" />
          <p className="text-sm font-medium">RAG system not available</p>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
          {/* Health Status */}
          {health && (
            <div className="rounded-2xl border border-slate-200 bg-white p-4">
              <div className="flex items-center gap-2 mb-3">
                {health.rag_initialized ? (
                  <CheckCircle2 size={16} className="text-emerald-500" />
                ) : (
                  <XCircle size={16} className="text-red-500" />
                )}
                <h3 className="text-sm font-medium text-slate-700">
                  System Health
                </h3>
                <span
                  className={`ml-auto px-2.5 py-0.5 rounded-full text-[10px] font-semibold tracking-wide ${
                    health.rag_initialized
                      ? "bg-emerald-50 text-emerald-600 border border-emerald-200"
                      : "bg-red-50 text-red-500 border border-red-200"
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
                    className="flex items-center gap-2 px-2.5 py-2 rounded-xl bg-slate-50 border border-slate-200"
                  >
                    <div className="relative">
                      <div
                        className={`w-2 h-2 rounded-full ${
                          comp.ok ? "bg-emerald-500" : "bg-red-400"
                        }`}
                      />
                      {comp.ok && (
                        <div className="absolute inset-0 w-2 h-2 rounded-full bg-emerald-400 animate-ping opacity-30" />
                      )}
                    </div>
                    <span className="text-[10px] text-slate-500">
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
              color="bg-indigo-50 text-indigo-600"
            />
            <StatCard
              icon={Network}
              label="Entities"
              value={stats.memories?.entities ?? 0}
              color="bg-purple-50 text-purple-600"
            />
            <StatCard
              icon={Database}
              label="Graph Edges"
              value={stats.memories?.edges ?? 0}
              color="bg-blue-50 text-blue-600"
            />
            <StatCard
              icon={Zap}
              label="Belief Deltas"
              value={stats.memories?.belief_deltas ?? 0}
              color="bg-amber-50 text-amber-600"
            />
          </div>

          {/* Vectors */}
          {stats.vectors && (
            <div className="rounded-2xl border border-slate-200 bg-white p-4">
              <h3 className="text-xs font-medium text-slate-500 mb-3 uppercase tracking-wider">
                Vector Store
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div>
                  <p className="text-lg font-bold text-slate-800">
                    {stats.vectors.total_vectors ?? 0}
                  </p>
                  <p className="text-[10px] text-slate-400">Total Vectors</p>
                </div>
                <div>
                  <p className="text-lg font-bold text-slate-800">
                    {stats.vectors.dimension ?? 0}
                  </p>
                  <p className="text-[10px] text-slate-400">Dimensions</p>
                </div>
                <div>
                  <p className="text-lg font-bold text-slate-800">
                    {stats.vectors.hot_count ?? 0}
                  </p>
                  <p className="text-[10px] text-slate-400">Hot Vectors</p>
                </div>
                <div>
                  <p className="text-lg font-bold text-slate-800">
                    {stats.vectors.cold_count ?? 0}
                  </p>
                  <p className="text-[10px] text-slate-400">Cold Vectors</p>
                </div>
              </div>
            </div>
          )}

          {/* Graph Stats */}
          {stats.graph && (
            <div className="rounded-2xl border border-slate-200 bg-white p-4">
              <h3 className="text-xs font-medium text-slate-500 mb-3 uppercase tracking-wider">
                Knowledge Graph
              </h3>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <p className="text-lg font-bold text-slate-800">
                    {stats.graph.nodes ?? 0}
                  </p>
                  <p className="text-[10px] text-slate-400">Nodes</p>
                </div>
                <div>
                  <p className="text-lg font-bold text-slate-800">
                    {stats.graph.edges ?? 0}
                  </p>
                  <p className="text-[10px] text-slate-400">Edges</p>
                </div>
              </div>
            </div>
          )}

          {/* Cache */}
          {stats.cache && (
            <div className="rounded-2xl border border-slate-200 bg-white p-4">
              <h3 className="text-xs font-medium text-slate-500 mb-3 uppercase tracking-wider">
                Cache Performance
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div>
                  <p className="text-lg font-bold text-emerald-600">
                    {((stats.cache.hit_rate ?? 0) * 100).toFixed(1)}%
                  </p>
                  <p className="text-[10px] text-slate-400">Hit Rate</p>
                </div>
                <div>
                  <p className="text-lg font-bold text-slate-800">
                    {stats.cache.total_hits ?? 0}
                  </p>
                  <p className="text-[10px] text-slate-400">Hits</p>
                </div>
                <div>
                  <p className="text-lg font-bold text-slate-800">
                    {stats.cache.total_queries ?? 0}
                  </p>
                  <p className="text-[10px] text-slate-400">Total Queries</p>
                </div>
                <div>
                  <p className="text-lg font-bold text-slate-800">
                    {(stats.cache.exact_cache_size ?? 0) + (stats.cache.semantic_cache_size ?? 0)}
                  </p>
                  <p className="text-[10px] text-slate-400">Cached Entries</p>
                </div>
              </div>
            </div>
          )}

          {/* LLM Stats */}
          {stats.llm && (
            <div className="rounded-2xl border border-slate-200 bg-white p-4">
              <h3 className="text-xs font-medium text-slate-500 mb-3 uppercase tracking-wider">
                LLM Usage
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                <div>
                  <p className="text-lg font-bold text-slate-800">
                    {stats.llm.call_count ?? 0}
                  </p>
                  <p className="text-[10px] text-slate-400">Total Calls</p>
                </div>
                <div>
                  <p className="text-lg font-bold text-slate-800">
                    {stats.llm.total_tokens ?? 0}
                  </p>
                  <p className="text-[10px] text-slate-400">Total Tokens</p>
                </div>
                <div>
                  <p className="text-lg font-bold text-slate-800">
                    {stats.llm.model_loaded ? "✓ Loaded" : "✗ Not loaded"}
                  </p>
                  <p className="text-[10px] text-slate-400">Model Status</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
