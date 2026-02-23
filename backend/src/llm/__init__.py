"""
Local LLM Interface for Cortex Lab — Fine-Tuned Model Integration
Interfaces with the fine-tuned DeepSeek-R1-7B model.
Leverages all 15 training stages:
  Stage 1: Faithfulness (grounded generation)
  Stage 2: Agentic routing (structured JSON intent classification)
  Stage 3: Causal reasoning
  Stage 4: Self-RAG critique tokens (ISREL/ISSUP/ISUSE)
  Stage 5: Belief evolution tracking
  Stage 6: Summarization
  Stage 7: Dialogue coherence
  Stage 8: Long-context handling
  Stage 9: DPO preference alignment
  Stage 10: User-style adaptation
  Stage 11: ORPO alignment
  Stage 12: RAFT distractor-aware generation
  Stage 13: Function calling / tool use
  Stage 14: Rejection fine-tuning (knowing limits)
  Stage 15: SPIN self-play improvement
"""

import time
import re
import json
from typing import Optional, Dict, List, Any
import torch

# Stop patterns that indicate the model is hallucinating new turns
_LLM_STOP_PATTERNS = [
    "\nUser:", "\nuser:", "\nHuman:", "\nhuman:",
    "\nQ:", "\nA:", "\n\nUser ", "\nQuestion:",
]


def _truncate_at_stop(text: str) -> str:
    """Truncate at the first occurrence of any stop pattern."""
    earliest = len(text)
    for pattern in _LLM_STOP_PATTERNS:
        pos = text.find(pattern)
        if pos != -1 and pos < earliest:
            earliest = pos
    return text[:earliest].strip()


class LocalLLM:
    """
    Interface to the fine-tuned DeepSeek-R1-7B model.
    Provides structured LLM calls leveraging fine-tuned capabilities:

    Core methods:
    - generate(): General text generation
    - classify(): Quick classification with constrained output
    - extract_json(): Parse JSON from model output
    - summarize(): Concise summarization (Stage 6)

    Fine-tuning-aware methods:
    - route_query(): Structured JSON routing (Stage 2 Agentic)
    - self_rag_critique(): ISREL/ISSUP/ISUSE token generation (Stage 4)
    - causal_reason(): Cause-effect chain analysis (Stage 3)
    - detect_belief_change(): Belief evolution tracking (Stage 5)
    - generate_faithful(): Grounded generation with citations (Stage 1)
    - call_function(): Tool/function calling (Stage 13)
    - raft_generate(): Distractor-aware generation (Stage 12)
    """

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self._call_count = 0
        self._total_tokens = 0
        self._total_time_ms = 0

    def set_model(self, model, tokenizer):
        """Set model reference (called after model loads in server.py)."""
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str, max_tokens: int = 512,
                 temperature: float = 0.3, top_p: float = 0.9,
                 stop_sequences: Optional[list] = None) -> str:
        """Generate text from the model with stop-pattern safety."""
        if self.model is None or self.tokenizer is None:
            return self._fallback_generate(prompt)

        t0 = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[-1]

        # Build stop token IDs
        eos_ids = [self.tokenizer.eos_token_id]
        for stop_str in ["User:", "<|im_end|>", "<|endoftext|>"]:
            try:
                ids = self.tokenizer.encode(stop_str, add_special_tokens=False)
                if ids:
                    eos_ids.append(ids[0])
            except Exception:
                pass

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, 1024),
                temperature=max(temperature, 0.01),
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=eos_ids,
                repetition_penalty=1.2,
            )

        generated = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()

        # Strip thinking tags if present
        if "<think>" in generated:
            think_end = generated.find("</think>")
            if think_end > -1:
                generated = generated[think_end + len("</think>"):].strip()

        # Apply custom stop sequences
        if stop_sequences:
            for seq in stop_sequences:
                if seq in generated:
                    generated = generated[:generated.index(seq)]

        # Always truncate at hallucinated conversation continuations
        generated = _truncate_at_stop(generated)

        elapsed_ms = (time.time() - t0) * 1000
        self._call_count += 1
        self._total_tokens += input_len + len(self.tokenizer.encode(generated))
        self._total_time_ms += elapsed_ms

        return generated

    def classify(self, prompt: str, options: list, default: str = "") -> str:
        """Quick classification with constrained output."""
        full_prompt = f"{prompt}\n\nChoose EXACTLY ONE from: {', '.join(options)}\nAnswer:"
        result = self.generate(full_prompt, max_tokens=20, temperature=0.1)
        result_lower = result.strip().lower()

        for opt in options:
            if opt.lower() in result_lower:
                return opt
        return default or options[0]

    def extract_json(self, prompt: str, max_tokens: int = 256) -> dict:
        """Generate and parse JSON output."""
        result = self.generate(prompt + "\n\nOutput valid JSON only:", max_tokens=max_tokens, temperature=0.1)
        try:
            start = result.find("{")
            end = result.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(result[start:end])
        except json.JSONDecodeError:
            pass
        return {}

    def summarize(self, text: str, max_length: int = 100) -> str:
        """Generate a concise summary (leverages Stage 6 training)."""
        prompt = f"""Summarize the following text concisely, preserving key entities, dates, and causal relationships.
Keep it under {max_length} words.

Text: {text}

Summary:"""
        return self.generate(prompt, max_tokens=max_length * 2, temperature=0.2)

    # ─── Fine-Tuning-Aware Methods ────────────────────────────────────────

    def route_query(self, query: str, session_context: str = "") -> Dict[str, Any]:
        """
        Stage 2 (Agentic Routing): Structured JSON intent classification.
        Returns: {intent, complexity, agents, reasoning, needs_retrieval}
        """
        prompt = f"""<|im_start|>system
You are Cortex Lab's query router. Analyze the user's query and output a JSON routing decision.

Available intents: temporal, causal, reflective, factual, procedural, comparative, exploratory
Available agents: timeline, causal, reflection, planning, arbitration
Complexity: low (0.0-0.3), medium (0.3-0.6), high (0.6-1.0)
<|im_end|>
<|im_start|>user
{f"Session context: {session_context[:200]}" if session_context else ""}
Query: {query}

Output routing decision as JSON:
<|im_end|>
<|im_start|>assistant
"""
        result = self.extract_json(prompt, max_tokens=200)

        # Validate and provide defaults
        defaults = {
            "intent": "exploratory",
            "complexity": 0.5,
            "agents": ["planning"],
            "needs_retrieval": True,
            "reasoning": "Default routing",
        }
        for k, v in defaults.items():
            if k not in result:
                result[k] = v

        return result

    def self_rag_critique(self, query: str, answer: str,
                          evidence: List[str]) -> Dict[str, Any]:
        """
        Stage 4 (Self-RAG): Generate ISREL/ISSUP/ISUSE critique tokens.
        Returns structured critique with scores.
        """
        evidence_text = "\n".join(f"[{i+1}] {e[:200]}" for i, e in enumerate(evidence[:5]))

        prompt = f"""<|im_start|>system
You are a retrieval quality evaluator. For the given query, answer, and evidence,
evaluate three criteria and provide scores:

ISREL (Is Relevant): Does the evidence address the query? Score 1-10.
ISSUP (Is Supported): Is the answer grounded in the evidence? Score 1-10.
ISUSE (Is Useful): Is the answer useful and complete for the user? Score 1-10.

Output JSON with scores and brief justifications.
<|im_end|>
<|im_start|>user
Query: {query}
Answer: {answer[:300]}

Evidence:
{evidence_text}
<|im_end|>
<|im_start|>assistant
"""
        result = self.extract_json(prompt, max_tokens=200)

        # Parse scores with fallbacks
        isrel = min(max(result.get("ISREL", result.get("isrel", result.get("relevance", 5))), 1), 10)
        issup = min(max(result.get("ISSUP", result.get("issup", result.get("support", 5))), 1), 10)
        isuse = min(max(result.get("ISUSE", result.get("isuse", result.get("usefulness", 5))), 1), 10)

        return {
            "ISREL": isrel,
            "ISSUP": issup,
            "ISUSE": isuse,
            "avg_score": (isrel + issup + isuse) / 3.0,
            "verdict": "ACCEPT" if (isrel + issup + isuse) / 3.0 >= 6.0 else "REVISE",
            "justification": result.get("justification", result.get("reasoning", "")),
        }

    def causal_reason(self, query: str, memories: List[str]) -> str:
        """
        Stage 3 (Causal Reasoning): Trace cause-effect chains from memories.
        """
        mem_text = "\n".join(f"[{i+1}] {m[:200]}" for i, m in enumerate(memories[:8]))

        prompt = f"""<|im_start|>system
You are Cortex Lab's causal reasoning engine. Analyze the user's memories to trace
cause-effect relationships. Structure your response as:
1. Identify the causal chain (what caused what)
2. Note any contributing factors
3. Describe the effects/outcomes
Be grounded in the evidence — never fabricate causal links.
<|im_end|>
<|im_start|>user
Query: {query}

Memories:
{mem_text}
<|im_end|>
<|im_start|>assistant
"""
        return self.generate(prompt, max_tokens=500, temperature=0.3)

    def detect_belief_change(self, old_text: str, new_text: str, topic: str = "") -> Dict[str, Any]:
        """
        Stage 5 (Belief Evolution): Detect stance changes between two memories.
        Returns: {change_type, old_stance, new_stance, confidence, explanation}
        """
        prompt = f"""<|im_start|>system
You are a belief evolution detector. Compare two memories about the same topic
and classify the change. Types: CONTRADICTION, REFINEMENT, EXPANSION, REINFORCEMENT, NONE.
Output JSON with: change_type, old_stance, new_stance, confidence (0-1), explanation.
<|im_end|>
<|im_start|>user
Topic: {topic or "general"}
Earlier memory: {old_text[:300]}
Later memory: {new_text[:300]}
<|im_end|>
<|im_start|>assistant
"""
        result = self.extract_json(prompt, max_tokens=200)
        defaults = {
            "change_type": "NONE",
            "old_stance": "",
            "new_stance": "",
            "confidence": 0.5,
            "explanation": "",
        }
        for k, v in defaults.items():
            if k not in result:
                result[k] = v
        return result

    def generate_faithful(self, query: str, evidence: List[str],
                          session_context: str = "") -> str:
        """
        Stage 1 (Faithfulness): Generate a grounded answer with inline citations.
        References evidence as [1], [2], etc.
        """
        evidence_text = "\n".join(f"[{i+1}] {e[:250]}" for i, e in enumerate(evidence[:5]))

        prompt = f"""<|im_start|>system
You are Cortex Lab, a personal AI memory assistant. Answer the user's question
based ONLY on the provided evidence from their memories. Use inline citations
like [1], [2] to reference specific memories. If the evidence is insufficient,
honestly say so — never fabricate information.
{f"Session context: {session_context[:200]}" if session_context else ""}
<|im_end|>
<|im_start|>user
{query}

Evidence:
{evidence_text}
<|im_end|>
<|im_start|>assistant
"""
        return self.generate(prompt, max_tokens=500, temperature=0.3)

    def raft_generate(self, query: str, oracle_docs: List[str],
                      distractor_docs: List[str]) -> str:
        """
        Stage 12 (RAFT): Generate answers while ignoring distractor documents.
        Trained to identify and use only relevant docs from mixed context.
        """
        # Interleave oracle and distractor documents
        all_docs = []
        for i, doc in enumerate(oracle_docs[:3]):
            all_docs.append(f"[Doc {len(all_docs)+1}] {doc[:200]}")
        for i, doc in enumerate(distractor_docs[:3]):
            all_docs.append(f"[Doc {len(all_docs)+1}] {doc[:200]}")

        # Shuffle to test distractor resistance (like RAFT training)
        import random
        random.shuffle(all_docs)
        docs_text = "\n".join(all_docs)

        prompt = f"""<|im_start|>system
You are Cortex Lab. Answer the question using ONLY the relevant documents.
Some documents may be distractors — ignore them. Cite relevant documents with [Doc N].
<|im_end|>
<|im_start|>user
{query}

Documents:
{docs_text}
<|im_end|>
<|im_start|>assistant
"""
        return self.generate(prompt, max_tokens=400, temperature=0.3)

    def call_function(self, query: str, available_tools: List[Dict]) -> Dict[str, Any]:
        """
        Stage 13 (Function Calling): Parse user intent into tool calls.
        Returns: {tool_name, arguments, reasoning}
        """
        tools_desc = "\n".join(
            f"- {t['name']}: {t.get('description', '')} | params: {json.dumps(t.get('parameters', {}))}"
            for t in available_tools
        )

        prompt = f"""<|im_start|>system
You are a function calling assistant. Given the user's request and available tools,
decide which tool to call and with what arguments.
Output JSON: {{"tool_name": "...", "arguments": {{...}}, "reasoning": "..."}}
If no tool is needed, set tool_name to "none".

Available tools:
{tools_desc}
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""
        result = self.extract_json(prompt, max_tokens=200)
        if "tool_name" not in result:
            result["tool_name"] = "none"
        return result

    # ─── Fallback & Stats ─────────────────────────────────────────────────

    def _fallback_generate(self, prompt: str) -> str:
        """Fallback when model is not loaded."""
        return "[Model not loaded - cannot generate response]"

    def get_stats(self) -> dict:
        return {
            "call_count": self._call_count,
            "total_tokens": self._total_tokens,
            "total_time_ms": round(self._total_time_ms, 1),
            "avg_latency_ms": round(self._total_time_ms / max(self._call_count, 1), 1),
            "model_loaded": self.model is not None,
        }

    def reset_stats(self):
        self._call_count = 0
        self._total_tokens = 0
        self._total_time_ms = 0
