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
      ctx.strokeStyle = "rgba(100, 116, 139, 0.2)";
      ctx.lineWidth = Math.max(1, (edge.weight || 0.5) * 2);
      ctx.stroke();

      // Edge label
      if (edge.relation) {
        const mx = (source.x + target.x) / 2;
        const my = (source.y + target.y) / 2;
        ctx.font = "9px Inter, sans-serif";
        ctx.fillStyle = "rgba(148, 163, 184, 0.5)";
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
      ctx.strokeStyle = isSelected ? "#ffffff" : color + "80";
      ctx.lineWidth = isSelected ? 2 : 1;
      ctx.stroke();

      // Label
      ctx.font = "11px Inter, sans-serif";
      ctx.fillStyle = "#e2e8f0";
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
            <Network size={18} className="text-deepseek-400" />
            <h2 className="text-sm font-semibold text-surface-200">
              Knowledge Graph
            </h2>
            {graphData && (
              <span className="text-[10px] text-surface-500 bg-surface-800/60 px-2 py-0.5 rounded-md">
                {graphData.nodes.length} nodes · {graphData.edges.length} edges
              </span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setZoom((z) => Math.min(5, z * 1.2))}
            className="p-1.5 rounded-lg text-surface-500 hover:text-surface-300 hover:bg-surface-800/50 transition-colors"
          >
            <ZoomIn size={16} />
          </button>
          <button
            onClick={() => setZoom((z) => Math.max(0.1, z * 0.8))}
            className="p-1.5 rounded-lg text-surface-500 hover:text-surface-300 hover:bg-surface-800/50 transition-colors"
          >
            <ZoomOut size={16} />
          </button>
          <button
            onClick={() => {
              setZoom(1);
              setPan({ x: 0, y: 0 });
            }}
            className="p-1.5 rounded-lg text-surface-500 hover:text-surface-300 hover:bg-surface-800/50 transition-colors"
          >
            <Maximize2 size={16} />
          </button>
          <button
            onClick={loadGraph}
            className="p-1.5 rounded-lg text-surface-500 hover:text-surface-300 hover:bg-surface-800/50 transition-colors"
          >
            <RefreshCw size={16} />
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
              <Loader2 size={24} className="animate-spin text-deepseek-400" />
            </div>
          ) : graphData && graphData.nodes.length === 0 ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-surface-500">
              <Network size={40} className="mb-3 opacity-30" />
              <p className="text-sm">No graph data yet</p>
              <p className="text-xs mt-1">
                Chat with Cortex Lab to build the knowledge graph
              </p>
            </div>
          ) : (
            <canvas ref={canvasRef} className="w-full h-full cursor-grab active:cursor-grabbing" />
          )}

          {/* Legend */}
          <div className="absolute bottom-3 left-3 bg-surface-900/90 backdrop-blur-sm border border-surface-800/50 rounded-lg p-3">
            <p className="text-[10px] text-surface-500 mb-1.5 font-medium">Node Types</p>
            <div className="flex flex-wrap gap-2">
              {Object.entries(typeColors)
                .filter(([k]) => k !== "default")
                .map(([type, color]) => (
                  <div key={type} className="flex items-center gap-1">
                    <div
                      className="w-2.5 h-2.5 rounded-full"
                      style={{ backgroundColor: color }}
                    />
                    <span className="text-[9px] text-surface-400 capitalize">
                      {type}
                    </span>
                  </div>
                ))}
            </div>
          </div>
        </div>

        {/* Node Details Panel */}
        {selectedNode && (
          <div className="w-64 border-l border-surface-800/50 bg-surface-900/50 p-4 space-y-3 overflow-y-auto">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-surface-200">
                {selectedNode.label}
              </h3>
              <button
                onClick={() => setSelectedNode(null)}
                className="text-surface-500 hover:text-surface-300 text-xs"
              >
                ✕
              </button>
            </div>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-surface-500">Type</span>
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
                  <span className="text-[10px] text-surface-500">Mentions</span>
                  <span className="text-xs text-surface-300">{selectedNode.mentions}</span>
                </div>
              )}
              {selectedNode.firstSeen && (
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-surface-500">First Seen</span>
                  <span className="text-xs text-surface-300">
                    {new Date(selectedNode.firstSeen).toLocaleDateString()}
                  </span>
                </div>
              )}
              {selectedNode.lastSeen && (
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-surface-500">Last Seen</span>
                  <span className="text-xs text-surface-300">
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
