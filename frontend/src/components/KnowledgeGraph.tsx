"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  ArrowLeft,
  Loader2,
  Network,
  ZoomIn,
  ZoomOut,
  Maximize2,
  RefreshCw,
} from "lucide-react";
import { GraphData, GraphNode, GraphEdge } from "@/lib/types";
import { getGraphData } from "@/lib/api";

interface NodePosition {
  x: number;
  y: number;
  vx: number;
  vy: number;
  node: GraphNode;
}

export function KnowledgeGraph({ onBack }: { onBack: () => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const nodesRef = useRef<NodePosition[]>([]);
  const animFrameRef = useRef<number>(0);
  const isDragging = useRef(false);
  const dragStart = useRef({ x: 0, y: 0 });
  const dragNode = useRef<NodePosition | null>(null);

  const typeColors: Record<string, string> = {
    person: "#10b981",
    location: "#3b82f6",
    organization: "#a855f7",
    concept: "#f59e0b",
    event: "#ef4444",
    object: "#06b6d4",
    default: "#64748b",
  };

  const loadGraph = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getGraphData();
      setGraphData(data);
      initializePositions(data);
    } catch (err) {
      console.error("Failed to load graph:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadGraph();
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [loadGraph]);

  const initializePositions = (data: GraphData) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const radius = Math.min(cx, cy) * 0.6;

    nodesRef.current = data.nodes.map((node, i) => {
      const angle = (2 * Math.PI * i) / data.nodes.length;
      return {
        x: cx + radius * Math.cos(angle) + (Math.random() - 0.5) * 40,
        y: cy + radius * Math.sin(angle) + (Math.random() - 0.5) * 40,
        vx: 0,
        vy: 0,
        node,
      };
    });
    startSimulation(data);
  };

  const startSimulation = (data: GraphData) => {
    let iteration = 0;
    const maxIterations = 300;

    const tick = () => {
      if (iteration >= maxIterations) {
        draw(data);
        return;
      }

      const alpha = 1 - iteration / maxIterations;
      const nodes = nodesRef.current;

      // Repulsion between all nodes
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[j].x - nodes[i].x;
          const dy = nodes[j].y - nodes[i].y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const force = (alpha * 2000) / (dist * dist);
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          nodes[i].vx -= fx;
          nodes[i].vy -= fy;
          nodes[j].vx += fx;
          nodes[j].vy += fy;
        }
      }

      // Attraction along edges
      const nodeIndex = new Map(nodes.map((n, i) => [n.node.id, i]));
      for (const edge of data.edges) {
        const si = nodeIndex.get(edge.source);
        const ti = nodeIndex.get(edge.target);
        if (si === undefined || ti === undefined) continue;
        const dx = nodes[ti].x - nodes[si].x;
        const dy = nodes[ti].y - nodes[si].y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = alpha * (dist - 120) * 0.01;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        nodes[si].vx += fx;
        nodes[si].vy += fy;
        nodes[ti].vx -= fx;
        nodes[ti].vy -= fy;
      }

      // Center gravity
      const canvas = canvasRef.current;
      if (canvas) {
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;
        for (const n of nodes) {
          n.vx += (cx - n.x) * alpha * 0.005;
          n.vy += (cy - n.y) * alpha * 0.005;
        }
      }

      // Apply velocity with damping
      for (const n of nodes) {
        if (dragNode.current === n) continue;
        n.vx *= 0.85;
        n.vy *= 0.85;
        n.x += n.vx;
        n.y += n.vy;
      }

      iteration++;
      draw(data);
      animFrameRef.current = requestAnimationFrame(tick);
    };

    tick();
  };

  const draw = (data: GraphData) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.translate(pan.x, pan.y);
    ctx.scale(zoom, zoom);

    const nodes = nodesRef.current;
    const nodeIndex = new Map(nodes.map((n) => [n.node.id, n]));

    // Draw edges
    for (const edge of data.edges) {
      const source = nodeIndex.get(edge.source);
      const target = nodeIndex.get(edge.target);
      if (!source || !target) continue;

      ctx.beginPath();
      ctx.moveTo(source.x, source.y);
      ctx.lineTo(target.x, target.y);
      ctx.strokeStyle = "rgba(100, 116, 139, 0.15)";
      ctx.lineWidth = Math.max(1, (edge.weight || 0.5) * 2);
      ctx.stroke();

      // Edge label
      if (edge.relation) {
        const mx = (source.x + target.x) / 2;
        const my = (source.y + target.y) / 2;
        ctx.font = "9px Inter, sans-serif";
        ctx.fillStyle = "rgba(100, 116, 139, 0.6)";
        ctx.textAlign = "center";
        ctx.fillText(edge.relation, mx, my - 4);
      }
    }

    // Draw nodes
    for (const np of nodes) {
      const { node, x, y } = np;
      const size = 6 + (node.mentions || 1) * 2;
      const color = typeColors[node.type] || typeColors.default;
      const isSelected = selectedNode?.id === node.id;

      // Glow
      if (isSelected) {
        ctx.beginPath();
        ctx.arc(x, y, size + 6, 0, Math.PI * 2);
        ctx.fillStyle = color + "30";
        ctx.fill();
      }

      // Circle
      ctx.beginPath();
      ctx.arc(x, y, size, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = isSelected ? "#4f46e5" : color + "80";
      ctx.lineWidth = isSelected ? 2 : 1;
      ctx.stroke();

      // Label
      ctx.font = "11px Inter, sans-serif";
      ctx.fillStyle = "#334155";
      ctx.textAlign = "center";
      ctx.fillText(node.label, x, y + size + 14);
    }

    ctx.restore();
  };

  // Handle canvas resize
  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const resize = () => {
      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
      if (graphData) draw(graphData);
    };

    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    return () => observer.disconnect();
  }, [graphData, zoom, pan, selectedNode]);

  // Mouse interactions
  const handleMouseDown = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left - pan.x) / zoom;
    const my = (e.clientY - rect.top - pan.y) / zoom;

    // Check if clicking a node
    for (const np of nodesRef.current) {
      const size = 6 + (np.node.mentions || 1) * 2;
      const dx = np.x - mx;
      const dy = np.y - my;
      if (dx * dx + dy * dy < size * size) {
        dragNode.current = np;
        setSelectedNode(np.node);
        return;
      }
    }

    isDragging.current = true;
    dragStart.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (dragNode.current) {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      dragNode.current.x = (e.clientX - rect.left - pan.x) / zoom;
      dragNode.current.y = (e.clientY - rect.top - pan.y) / zoom;
      if (graphData) draw(graphData);
    } else if (isDragging.current) {
      setPan({
        x: e.clientX - dragStart.current.x,
        y: e.clientY - dragStart.current.y,
      });
    }
  };

  const handleMouseUp = () => {
    isDragging.current = false;
    dragNode.current = null;
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom((z) => Math.max(0.1, Math.min(5, z * delta)));
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
            <Network size={18} className="text-indigo-500" />
            <h2 className="text-sm font-semibold text-slate-700">
              Knowledge Graph
            </h2>
            {graphData && (
              <span className="text-[10px] text-slate-500 bg-slate-100 border border-slate-200 px-2 py-0.5 rounded-lg">
                {graphData.nodes.length} nodes · {graphData.edges.length} edges
              </span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setZoom((z) => Math.min(5, z * 1.2))}
            className="p-2 rounded-xl text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-all duration-200"
          >
            <ZoomIn size={15} />
          </button>
          <button
            onClick={() => setZoom((z) => Math.max(0.1, z * 0.8))}
            className="p-2 rounded-xl text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-all duration-200"
          >
            <ZoomOut size={15} />
          </button>
          <button
            onClick={() => {
              setZoom(1);
              setPan({ x: 0, y: 0 });
            }}
            className="p-2 rounded-xl text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-all duration-200"
          >
            <Maximize2 size={15} />
          </button>
          <button
            onClick={loadGraph}
            className="p-2 rounded-xl text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-all duration-200"
          >
            <RefreshCw size={15} />
          </button>
        </div>
      </div>

      {/* Graph */}
      <div className="flex flex-1 overflow-hidden">
        <div
          ref={containerRef}
          className="flex-1 relative"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onWheel={handleWheel}
        >
          {loading ? (
            <div className="absolute inset-0 flex items-center justify-center">
              <Loader2 size={24} className="animate-spin text-indigo-500" />
            </div>
          ) : graphData && graphData.nodes.length === 0 ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-400 fade-in">
              <Network size={40} className="mb-3 opacity-30" />
              <p className="text-sm font-medium">No graph data yet</p>
              <p className="text-xs mt-1 text-slate-400">
                Chat with Cortex Lab to build the knowledge graph
              </p>
            </div>
          ) : (
            <canvas ref={canvasRef} className="w-full h-full cursor-grab active:cursor-grabbing bg-slate-50" />
          )}

          {/* Legend */}
          <div className="absolute bottom-3 left-3 bg-white/80 backdrop-blur-xl rounded-xl border border-slate-200 p-3 shadow-lg">
            <p className="text-[10px] text-slate-500 mb-2 font-semibold tracking-wider uppercase">Node Types</p>
            <div className="flex flex-wrap gap-2">
              {Object.entries(typeColors)
                .filter(([k]) => k !== "default")
                .map(([type, color]) => (
                  <div key={type} className="flex items-center gap-1">
                    <div
                      className="w-2.5 h-2.5 rounded-full"
                      style={{ backgroundColor: color }}
                    />
                    <span className="text-[9px] text-slate-500 capitalize">
                      {type}
                    </span>
                  </div>
                ))}
            </div>
          </div>
        </div>

        {/* Node Details Panel */}
        {selectedNode && (
          <div className="w-64 border-l border-slate-200 bg-white/90 backdrop-blur-xl p-4 space-y-3 overflow-y-auto fade-in">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-800">
                {selectedNode.label}
              </h3>
              <button
                onClick={() => setSelectedNode(null)}
                className="text-slate-400 hover:text-slate-600 text-xs transition-colors"
              >
                ✕
              </button>
            </div>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-slate-400">Type</span>
                <span
                  className="px-1.5 py-0.5 rounded text-[9px] font-medium capitalize"
                  style={{
                    backgroundColor: (typeColors[selectedNode.type] || typeColors.default) + "20",
                    color: typeColors[selectedNode.type] || typeColors.default,
                  }}
                >
                  {selectedNode.type}
                </span>
              </div>
              {selectedNode.mentions && (
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-slate-400">Mentions</span>
                  <span className="text-xs text-slate-600">{selectedNode.mentions}</span>
                </div>
              )}
              {selectedNode.firstSeen && (
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-slate-400">First Seen</span>
                  <span className="text-xs text-slate-600">
                    {new Date(selectedNode.firstSeen).toLocaleDateString()}
                  </span>
                </div>
              )}
              {selectedNode.lastSeen && (
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-slate-400">Last Seen</span>
                  <span className="text-xs text-slate-600">
                    {new Date(selectedNode.lastSeen).toLocaleDateString()}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
