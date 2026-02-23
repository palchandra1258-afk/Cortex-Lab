import { ChatMessage, ChatSettings, DEFAULT_SETTINGS, MemoryObject, GraphData, RAGStats, EvidenceCard } from "./types";

const API_BASE = "/api";

// ── Non-streaming chat ──────────────────────────────────────────

export async function sendMessage(
  messages: { role: string; content: string }[],
  settings: ChatSettings = DEFAULT_SETTINGS,
): Promise<{
  content: string;
  thinking?: string;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages,
      temperature: settings.temperature,
      top_p: settings.topP,
      max_tokens: settings.maxTokens,
      stream: false,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Server error ${res.status}`);
  }

  return res.json();
}

// ── Streaming chat ──────────────────────────────────────────────

export async function streamMessage(
  messages: { role: string; content: string }[],
  settings: ChatSettings = DEFAULT_SETTINGS,
  onToken: (token: string) => void,
  onDone: () => void,
  onError: (err: Error) => void,
): Promise<void> {
  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages,
        temperature: settings.temperature,
        top_p: settings.topP,
        max_tokens: settings.maxTokens,
        stream: true,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const reader = res.body?.getReader();
    if (!reader) throw new Error("No response body");

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data: ")) continue;

        const json = trimmed.slice(6);
        try {
          const data = JSON.parse(json);
          if (data.done) {
            onDone();
            return;
          }
          if (data.delta) {
            onToken(data.delta);
          }
        } catch {
          // skip malformed JSON
        }
      }
    }

    onDone();
  } catch (err) {
    onError(err instanceof Error ? err : new Error(String(err)));
  }
}

// ── RAG-Enhanced Chat ───────────────────────────────────────────

export async function ragChat(
  messages: { role: string; content: string }[],
  settings: ChatSettings = DEFAULT_SETTINGS,
  sessionId: string = "",
): Promise<{
  content: string;
  thinking?: string;
  evidence?: EvidenceCard[];
  agents_used?: string[];
  confidence?: number;
  query_analysis?: { intent: string; complexity: number; routing: string };
  processing_time_ms?: number;
  cache_hit?: boolean;
}> {
  const res = await fetch(`${API_BASE}/rag/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages,
      temperature: settings.temperature,
      top_p: settings.topP,
      max_tokens: settings.maxTokens,
      stream: false,
      use_rag: true,
      session_id: sessionId,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Server error ${res.status}`);
  }

  return res.json();
}

// ── RAG Streaming Chat ──────────────────────────────────────────

export interface RAGStreamMeta {
  evidence?: EvidenceCard[];
  agents_used?: string[];
  confidence?: number;
  query_analysis?: { intent: string; complexity: number; routing: string };
  thinking?: string;
}

export async function streamRAGMessage(
  messages: { role: string; content: string }[],
  settings: ChatSettings = DEFAULT_SETTINGS,
  sessionId: string = "",
  onMeta: (meta: RAGStreamMeta) => void,
  onToken: (token: string) => void,
  onDone: () => void,
  onError: (err: Error) => void,
): Promise<void> {
  try {
    const res = await fetch(`${API_BASE}/rag/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages,
        temperature: settings.temperature,
        top_p: settings.topP,
        max_tokens: settings.maxTokens,
        stream: true,
        use_rag: true,
        session_id: sessionId,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const reader = res.body?.getReader();
    if (!reader) throw new Error("No response body");

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data: ")) continue;

        const jsonStr = trimmed.slice(6);
        try {
          const data = JSON.parse(jsonStr);
          // Check for RAG metadata chunk
          if (data.rag_meta) {
            onMeta(data.rag_meta);
          }
          if (data.done) {
            onDone();
            return;
          }
          if (data.delta) {
            onToken(data.delta);
          }
        } catch {
          // skip malformed JSON
        }
      }
    }

    onDone();
  } catch (err) {
    onError(err instanceof Error ? err : new Error(String(err)));
  }
}

// ── Memory Management ───────────────────────────────────────────

export async function getMemories(
  limit: number = 50,
  offset: number = 0,
): Promise<{ memories: MemoryObject[]; total: number }> {
  const res = await fetch(`${API_BASE}/memories?limit=${limit}&offset=${offset}`);
  if (!res.ok) throw new Error(`Failed to fetch memories: ${res.status}`);
  return res.json();
}

export async function ingestMemory(
  content: string,
  source: string = "manual",
): Promise<{ status: string; memory: MemoryObject }> {
  const res = await fetch(`${API_BASE}/memories/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content, source }),
  });
  if (!res.ok) throw new Error(`Failed to ingest memory: ${res.status}`);
  return res.json();
}

export async function searchMemories(
  query: string,
  topK: number = 10,
): Promise<{ results: MemoryObject[]; count: number }> {
  const res = await fetch(`${API_BASE}/memories/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
  });
  if (!res.ok) throw new Error(`Failed to search memories: ${res.status}`);
  return res.json();
}

export async function deleteMemory(memoryId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/memories/${memoryId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Failed to delete memory: ${res.status}`);
}

// ── Knowledge Graph ─────────────────────────────────────────────

export async function getGraphData(): Promise<GraphData> {
  const res = await fetch(`${API_BASE}/graph`);
  if (!res.ok) throw new Error(`Failed to fetch graph: ${res.status}`);
  return res.json();
}

export async function getEntities(): Promise<{ entities: Record<string, unknown>[] }> {
  const res = await fetch(`${API_BASE}/entities`);
  if (!res.ok) throw new Error(`Failed to fetch entities: ${res.status}`);
  return res.json();
}

// ── Belief Evolution ────────────────────────────────────────────

export async function getBeliefDeltas(
  limit: number = 50,
): Promise<{
  beliefs: {
    id: string;
    topic: string;
    old_belief_text: string;
    new_belief_text: string;
    change_type: string;
    confidence: number;
    detected_at: string;
  }[];
}> {
  const res = await fetch(`${API_BASE}/beliefs?limit=${limit}`);
  if (!res.ok) throw new Error(`Failed to fetch beliefs: ${res.status}`);
  return res.json();
}

// ── GraphRAG Communities ────────────────────────────────────────

export async function getCommunities(): Promise<{
  communities: {
    community_id: number;
    members: string[];
    size: number;
    memory_count: number;
  }[];
}> {
  const res = await fetch(`${API_BASE}/communities`);
  if (!res.ok) throw new Error(`Failed to fetch communities: ${res.status}`);
  return res.json();
}

// ── RAG Stats ───────────────────────────────────────────────────

export async function getRAGStats(): Promise<RAGStats> {
  const res = await fetch(`${API_BASE}/rag/stats`);
  if (!res.ok) throw new Error(`Failed to fetch stats: ${res.status}`);
  return res.json();
}

export async function getRAGHealth(): Promise<{
  rag_initialized: boolean;
  stats: RAGStats;
}> {
  const res = await fetch(`${API_BASE}/rag/health`);
  if (!res.ok) throw new Error(`Failed to fetch RAG health: ${res.status}`);
  return res.json();
}
