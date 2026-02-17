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
  };
}

export interface ChatSettings {
  temperature: number;
  topP: number;
  maxTokens: number;
  stream: boolean;
}

export const DEFAULT_SETTINGS: ChatSettings = {
  temperature: 0.6,
  topP: 0.95,
  maxTokens: 2048,
  stream: true,
};
