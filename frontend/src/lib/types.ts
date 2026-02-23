// ── Shared types ────────────────────────────────────────────────

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  thinking?: string;
  timestamp: number;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  isStreaming?: boolean;
  // RAG-enhanced fields
  evidence?: EvidenceCard[];
  agentsUsed?: string[];
  confidence?: number;
  queryAnalysis?: QueryAnalysis;
  processingTimeMs?: number;
  cacheHit?: boolean;
}

export interface EvidenceCard {
  content: string;
  score: number;
  channel: string;
  timestamp: string;
  memory_type: string;
  emotion: string;
  entities: string[];
}

export interface QueryAnalysis {
  intent: string;
  complexity: number;
  routing: string;
}

export interface MemoryObject {
  id: string;
  content: string;
  memory_type: string;
  timestamp: string;
  emotion: string;
  emotion_confidence: number;
  importance: number;
  topics: string[];
  entities: string[];
  propositions: string[];
  source: string;
  score?: number;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface GraphNode {
  id: string;
  label: string;
  type: string;
  memory_count: number;
  mentions?: number;
  firstSeen?: string;
  lastSeen?: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  relation: string;
  weight: number;
}

export interface RAGStats {
  status: string;
  memories: {
    memories: number;
    entities: number;
    edges: number;
    belief_deltas: number;
    conversations: number;
    backend: string;
  };
  vectors: {
    total_vectors: number;
    hot_count: number;
    warm_count: number;
    cold_count: number;
    using_faiss: boolean;
    dimension: number;
  };
  graph: {
    nodes: number;
    edges: number;
    density?: number;
  };
  cache: {
    exact_hits: number;
    exact_misses: number;
    semantic_hits: number;
    semantic_misses: number;
    embedding_hits: number;
    embedding_misses: number;
    total_queries: number;
    total_hits: number;
    hit_rate: number;
    exact_cache_size: number;
    semantic_cache_size: number;
    embedding_cache_size: number;
  };
  llm: {
    call_count: number;
    total_tokens: number;
    model_loaded: boolean;
  };
}

export interface ModelStatus {
  status: string;
  model_loaded: boolean;
  model_info: {
    name?: string;
    parameters?: string;
    quantization?: string;
    device?: string;
    gpu_memory?: string;
    max_context?: number;
    load_time_seconds?: number;
    fine_tuned?: boolean;
    training_stages_completed?: number;
    base_model?: string;
  };
}

export interface ChatSettings {
  temperature: number;
  topP: number;
  maxTokens: number;
  stream: boolean;
  useRAG: boolean;
}

export const DEFAULT_SETTINGS: ChatSettings = {
  temperature: 0.6,
  topP: 0.95,
  maxTokens: 2048,
  stream: true,
  useRAG: true,
};
