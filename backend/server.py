"""
FastAPI Backend Server for Cortex Lab — Fine-Tuned DeepSeek-R1-7B
Serves the 15-stage curriculum fine-tuned model via REST API + Server-Sent Events (streaming).
Includes full Agentic RAG system with memory, retrieval, and multi-agent reasoning.

Model: DeepSeek-R1-Distill-Qwen-7B fine-tuned across 15 stages:
  Faithfulness → Agentic → Causal → Self-RAG → Belief → Summarization →
  Dialogue → LongContext → DPO → UserStyle → ORPO → RAFT → FunctionCalling →
  RFT → SPIN
"""

import os
import sys
import time
import json
import asyncio
import uuid
import traceback
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from threading import Thread

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Add backend dir to path for src imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.engine import rag_engine

# ── Configuration ────────────────────────────────────────────────────────────

# Fine-tuned model path — auto-detect latest merged stage
FINE_TUNED_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fine_tuned")

def _find_latest_merged_model():
    """Find the latest merged model from our fine-tuning pipeline."""
    # Check stages in reverse order (15 → 1) for the latest merged model
    stage_names = [
        "stage15_spin", "stage14_rft", "stage13_function_calling",
        "stage12_raft", "stage11_orpo", "stage10_user_style",
        "stage9_dpo", "stage8_longcontext", "stage7_dialogue",
        "stage6_summarization", "stage5_belief", "stage4_selfrag",
        "stage3_causal", "stage2_agentic", "stage1_faithfulness",
    ]
    for stage in stage_names:
        merged_path = os.path.join(FINE_TUNED_BASE, stage, "merged")
        if os.path.exists(merged_path) and os.path.exists(os.path.join(merged_path, "config.json")):
            print(f"  🎯 Found fine-tuned model: {stage}/merged")
            return merged_path
    return None

_fine_tuned_path = _find_latest_merged_model()
MODEL_NAME = os.environ.get("MODEL_NAME",
    _fine_tuned_path if _fine_tuned_path else "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

def _count_completed_stages():
    """Count how many training stages have completed."""
    count = 0
    for i in range(1, 16):
        stage_dirs = [d for d in os.listdir(FINE_TUNED_BASE) if d.startswith(f"stage{i}_")]
        for sd in stage_dirs:
            meta = os.path.join(FINE_TUNED_BASE, sd, "training_meta.json")
            if os.path.exists(meta):
                count += 1
    return count

USE_4BIT   = os.environ.get("USE_4BIT", "true").lower() == "true"   # Default ON for 7B
USE_8BIT   = os.environ.get("USE_8BIT", "false").lower() == "true"
HOST       = os.environ.get("HOST", "0.0.0.0")
PORT       = int(os.environ.get("PORT", "8000"))

# ── Global state ─────────────────────────────────────────────────────────────

model = None
tokenizer = None
model_loaded = False
model_info = {}

# ── Lifespan – loads model once on startup ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, model_loaded, model_info

    print("\n" + "=" * 64)
    print("  Cortex Lab  ·  Fine-Tuned DeepSeek-R1-7B  ·  FastAPI Backend")
    print("=" * 64 + "\n")

    # ── Tokenizer ────────────────────────────────────────────────────────
    print(f"[1/2] Loading tokenizer from: {MODEL_NAME[:80]}…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  ✓ Tokenizer ready")

    # ── Model ────────────────────────────────────────────────────────────
    # Help PyTorch manage GPU memory more efficiently
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    load_kwargs = {"trust_remote_code": True}

    if USE_4BIT:
        print("[2/2] Loading model in 4-bit with CPU offloading …")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        # Set max memory to leave some GPU memory free
        max_memory = {0: "17GB", "cpu": "30GB"}
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = max_memory
        load_kwargs["low_cpu_mem_usage"] = True
        load_kwargs["dtype"] = torch.bfloat16
        load_kwargs["offload_folder"] = "offload"
    elif USE_8BIT:
        print("[2/2] Loading model in 8-bit …")
        load_kwargs["load_in_8bit"] = True
        load_kwargs["device_map"] = "auto"
        load_kwargs["low_cpu_mem_usage"] = True
    else:
        print("[2/2] Loading model in full precision …")
        if torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float32

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
    elapsed = time.time() - t0

    if not (USE_4BIT or USE_8BIT):
        model.eval()

    model_loaded = True
    quant = "4-bit" if USE_4BIT else ("8-bit" if USE_8BIT else "fp16/fp32")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    gpu_mem  = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "N/A"

    completed_stages = _count_completed_stages()
    model_display_name = "DeepSeek-R1-7B (Fine-Tuned)" if _fine_tuned_path else "DeepSeek-R1-Distill-Qwen-7B"

    model_info = {
        "name": model_display_name,
        "parameters": "7B",
        "quantization": quant,
        "device": gpu_name,
        "gpu_memory": gpu_mem,
        "max_context": 32768,
        "load_time_seconds": round(elapsed, 1),
        "fine_tuned": _fine_tuned_path is not None,
        "training_stages_completed": completed_stages,
        "model_path": MODEL_NAME[:80],
        "base_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    }

    print(f"  ✓ Model loaded in {elapsed:.1f}s  ({quant} on {gpu_name})")
    print(f"  ✓ Fine-tuned: {_fine_tuned_path is not None} ({completed_stages}/15 stages)")
    print(f"\n  Server ready → http://{HOST}:{PORT}\n")

    # ── Initialize RAG Engine ────────────────────────────────────────────
    try:
        rag_engine.init(model=model, tokenizer=tokenizer)
    except Exception as e:
        print(f"  ⚠ RAG Engine initialization error: {e}")
        print("  ⚠ RAG features will be unavailable, basic chat still works.")

    yield  # ← app runs here

    # cleanup
    rag_engine.shutdown()
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Cortex Lab — Fine-Tuned DeepSeek-R1-7B Agentic RAG API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ──────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = Field(0.6, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(2048, ge=1, le=32768)
    stream: bool = False

class ChatResponse(BaseModel):
    id: str
    model: str
    created: int
    content: str
    thinking: Optional[str] = None
    usage: dict

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: dict

# ── RAG Schemas ──────────────────────────────────────────────────────────────

class RAGChatRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = Field(0.6, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(2048, ge=1, le=32768)
    stream: bool = False
    use_rag: bool = True  # Enable/disable RAG enhancement
    session_id: str = ""

class MemoryIngestRequest(BaseModel):
    content: str
    source: str = "manual"
    session_id: str = ""

class MemorySearchRequest(BaseModel):
    query: str
    top_k: int = 10

class ModelInfoResponse(BaseModel):
    """Detailed model information for the frontend."""
    name: str
    parameters: str
    quantization: str
    device: str
    gpu_memory: str
    max_context: int
    load_time_seconds: float
    fine_tuned: bool
    training_stages_completed: int
    base_model: str

# ── Helpers ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are Cortex Lab, a personal AI memory and reasoning assistant. "
    "You help the user by answering their questions thoughtfully and concisely. "
    "If the user asks about personal information (their name, preferences, etc.) "
    "that you don't actually know, honestly say you don't have that information yet "
    "and suggest they can teach you by telling you. "
    "Never fabricate personal details about the user. "
    "Keep responses focused and do NOT generate follow-up questions or continue "
    "the conversation on behalf of the user."
)

# Stop patterns: if the model starts generating these, it's hallucinating a new turn
_STOP_PATTERNS = ["\nUser:", "\nuser:", "\nHuman:", "\nhuman:", "\nQ:", "\nA:", "\n\nUser "]


def _build_prompt(messages: list[ChatMessage]) -> str:
    """
    Build a prompt using the tokenizer's chat template when available.
    Falls back to a structured format with system prompt and stop boundaries.
    """
    # Try to use the model's native chat template (best for DeepSeek-R1)
    if tokenizer is not None:
        try:
            chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for m in messages:
                chat_messages.append({"role": m.role, "content": m.content})
            prompt = tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
            return prompt
        except Exception:
            pass  # Fall back to manual format

    # Fallback: structured prompt with clear boundaries
    parts: list[str] = [f"System: {SYSTEM_PROMPT}"]
    for m in messages:
        if m.role == "user":
            parts.append(f"User: {m.content}")
        elif m.role == "assistant":
            parts.append(f"Assistant: {m.content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def _truncate_at_stop_patterns(text: str) -> str:
    """
    Truncate generated text at the first occurrence of any stop pattern.
    This prevents the model from hallucinating new conversation turns.
    """
    earliest_pos = len(text)
    for pattern in _STOP_PATTERNS:
        pos = text.find(pattern)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos
    return text[:earliest_pos].strip()


def _split_thinking(text: str):
    """
    Separate <think>…</think> reasoning from visible answer.
    Works with both raw special-token output and clean text.
    DeepSeek-R1 format: generation starts with <think>\n...reasoning...</think>answer
    """
    thinking = None
    content  = text

    # The generation prompt already contains <think>\n so output starts with thinking content
    if "<think>" in text:
        start = text.index("<think>") + len("<think>")
        if "</think>" in text:
            end = text.index("</think>")
            thinking = text[start:end].strip()
            content  = text[end + len("</think>"):].strip()
        else:
            # Model never closed the think tag — everything is thinking, no content
            thinking = text[start:].strip()
            content  = ""
    elif "</think>" in text:
        # Generation started inside <think> (prompt already had <think>\n)
        end = text.index("</think>")
        thinking = text[:end].strip()
        content  = text[end + len("</think>"):].strip()

    # Truncate hallucinated continuations from the visible content
    if content:
        content = _truncate_at_stop_patterns(content)
    return thinking, content

# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if model_loaded else "loading",
        model_loaded=model_loaded,
        model_info=model_info,
    )


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not model_loaded:
        raise HTTPException(503, "Model is still loading. Please wait.")

    prompt = _build_prompt(req.messages)

    # ── Streaming ────────────────────────────────────────────────────────
    if req.stream:
        return StreamingResponse(
            _stream_generate(prompt, req),
            media_type="text/event-stream",
        )

    # ── Non-streaming ────────────────────────────────────────────────────
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[-1]

    # Build stop token IDs to prevent runaway generation
    stop_token_ids = [tokenizer.eos_token_id]
    # Try to add common stop tokens
    for stop_str in ["User:", "<|im_end|>", "<|endoftext|>"]:
        try:
            ids = tokenizer.encode(stop_str, add_special_tokens=False)
            if ids:
                stop_token_ids.append(ids[0])
        except Exception:
            pass

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=min(req.max_tokens, 1024),
            temperature=max(req.temperature, 0.01),
            top_p=req.top_p,
            do_sample=req.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_token_ids,
            repetition_penalty=1.2,
        )

    # Decode with special tokens to extract <think>...</think>
    raw_output = tokenizer.decode(out[0][input_len:], skip_special_tokens=False).strip()
    thinking, content = _split_thinking(raw_output)

    # Clean up special tokens from content
    if content:
        # Remove any remaining special tokens
        for tok in ["<｜end▁of▁sentence｜>", "<|im_end|>", "<|endoftext|>", "<｜User｜>", "<｜Assistant｜>"]:
            content = content.replace(tok, "")
        content = _truncate_at_stop_patterns(content.strip())
    if thinking:
        for tok in ["<｜end▁of▁sentence｜>", "<|im_end|>", "<|endoftext|>", "<｜User｜>", "<｜Assistant｜>"]:
            thinking = thinking.replace(tok, "")
        thinking = thinking.strip()

    return ChatResponse(
        id=f"msg-{uuid.uuid4().hex[:12]}",
        model=model_info.get("name", MODEL_NAME),
        created=int(time.time()),
        content=content or "I'm not sure how to respond to that.",
        thinking=thinking,
        usage={
            "prompt_tokens": input_len,
            "completion_tokens": out.shape[-1] - input_len,
            "total_tokens": out.shape[-1],
        },
    )


async def _stream_generate(prompt: str, req: ChatRequest):
    """Yield Server-Sent Events token by token with stop-pattern detection."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Build stop token IDs
    stop_token_ids = [tokenizer.eos_token_id]
    for stop_str in ["User:", "<|im_end|>", "<|endoftext|>"]:
        try:
            ids = tokenizer.encode(stop_str, add_special_tokens=False)
            if ids:
                stop_token_ids.append(ids[0])
        except Exception:
            pass

    gen_kwargs = {
        **inputs,
        "max_new_tokens": min(req.max_tokens, 1024),
        "temperature": max(req.temperature, 0.01),
        "top_p": req.top_p,
        "do_sample": req.temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": stop_token_ids,
        "repetition_penalty": 1.2,
        "streamer": streamer,
    }

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    msg_id = f"msg-{uuid.uuid4().hex[:12]}"
    accumulated = ""  # Track full text to detect stop patterns mid-stream

    for token_text in streamer:
        accumulated += token_text
        # Check if we've hit a stop pattern in the accumulated text
        should_stop = False
        for pattern in _STOP_PATTERNS:
            if pattern in accumulated:
                # Send only the part before the stop pattern
                safe_part = accumulated[:accumulated.index(pattern)]
                leftover = safe_part[len(accumulated) - len(token_text):]
                if leftover:
                    chunk = {"id": msg_id, "delta": leftover}
                    yield f"data: {json.dumps(chunk)}\n\n"
                should_stop = True
                break
        if should_stop:
            break
        chunk = {"id": msg_id, "delta": token_text}
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0)

    yield f"data: {json.dumps({'id': msg_id, 'delta': '', 'done': True})}\n\n"
    thread.join()


# ── RAG-Enhanced Chat ────────────────────────────────────────────────────────

@app.post("/api/rag/chat")
async def rag_chat(req: RAGChatRequest):
    """RAG-enhanced chat: uses memory retrieval + multi-agent reasoning.
    Supports both streaming and non-streaming modes."""
    if not model_loaded:
        raise HTTPException(503, "Model is still loading.")
    if not rag_engine.initialized:
        raise HTTPException(503, "RAG engine is still initializing.")

    user_message = req.messages[-1].content if req.messages else ""
    if not user_message:
        raise HTTPException(400, "No message provided.")

    history = [{"role": m.role, "content": m.content} for m in req.messages[:-1]]

    # ── Streaming RAG ────────────────────────────────────────────────────
    if req.stream:
        return StreamingResponse(
            _stream_rag_generate(user_message, history, req),
            media_type="text/event-stream",
        )

    # ── Non-streaming RAG ────────────────────────────────────────────────
    try:
        result = await rag_engine.rag_chat(
            user_message=user_message,
            session_id=req.session_id,
            conversation_history=history,
        )

        return {
            "id": f"rag-{uuid.uuid4().hex[:12]}",
            "model": model_info.get("name", MODEL_NAME),
            "created": int(time.time()),
            "content": result.get("answer", ""),
            "thinking": result.get("thinking", ""),
            "evidence": result.get("evidence", []),
            "agents_used": result.get("agents_used", []),
            "confidence": result.get("confidence", 0),
            "query_analysis": result.get("query_analysis", {}),
            "processing_time_ms": result.get("processing_time_ms", 0),
            "cache_hit": result.get("cache_hit", False),
        }
    except Exception as e:
        print(f"  ❌ RAG error: {e}")
        traceback.print_exc()
        raise HTTPException(500, f"RAG processing error: {str(e)}")


async def _stream_rag_generate(user_message: str, history: list, req: RAGChatRequest):
    """
    Stream RAG-enhanced chat.
    1. Run RAG pipeline to get evidence + thinking (non-streamed)
    2. Stream the final answer generation token by token with evidence context
    """
    msg_id = f"rag-{uuid.uuid4().hex[:12]}"

    try:
        # Step 1: Run RAG pipeline for evidence retrieval (fast, no generation)
        rag_result = await rag_engine.rag_retrieve(
            user_message=user_message,
            session_id=req.session_id,
            conversation_history=history,
        )

        evidence = rag_result.get("evidence", [])
        agents_used = rag_result.get("agents_used", [])
        confidence = rag_result.get("confidence", 0)
        query_analysis = rag_result.get("query_analysis", {})
        thinking = rag_result.get("thinking", "")

        # Send metadata first
        meta_chunk = {
            "id": msg_id,
            "delta": "",
            "rag_meta": {
                "evidence": evidence,
                "agents_used": agents_used,
                "confidence": confidence,
                "query_analysis": query_analysis,
                "thinking": thinking,
            }
        }
        yield f"data: {json.dumps(meta_chunk)}\n\n"

        # Step 2: Build prompt with evidence context for streaming generation
        evidence_texts = [e.get("content", "")[:250] for e in evidence[:5]]
        evidence_block = "\n".join(f"[{i+1}] {e}" for i, e in enumerate(evidence_texts))

        rag_prompt = f"""<|im_start|>system
You are Cortex Lab, a personal AI memory and reasoning assistant.
Answer the user's question using the provided evidence from their memories.
Use inline citations [1], [2] etc. to reference specific memories.
If the evidence is insufficient, honestly say so — never fabricate information.
Keep responses focused and concise.
<|im_end|>
<|im_start|>user
{user_message}

Evidence from memories:
{evidence_block if evidence_block else "No relevant memories found."}
<|im_end|>
<|im_start|>assistant
"""

        inputs = tokenizer(rag_prompt, return_tensors="pt", truncation=True, max_length=4096)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        stop_token_ids = [tokenizer.eos_token_id]
        for stop_str in ["User:", "<|im_end|>", "<|endoftext|>"]:
            try:
                ids = tokenizer.encode(stop_str, add_special_tokens=False)
                if ids:
                    stop_token_ids.append(ids[0])
            except Exception:
                pass

        gen_kwargs = {
            **inputs,
            "max_new_tokens": min(req.max_tokens, 1024),
            "temperature": max(req.temperature, 0.01),
            "top_p": req.top_p,
            "do_sample": req.temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": stop_token_ids,
            "repetition_penalty": 1.2,
            "streamer": streamer,
        }

        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        accumulated = ""
        for token_text in streamer:
            accumulated += token_text
            should_stop = False
            for pattern in _STOP_PATTERNS:
                if pattern in accumulated:
                    safe_part = accumulated[:accumulated.index(pattern)]
                    leftover = safe_part[len(accumulated) - len(token_text):]
                    if leftover:
                        chunk = {"id": msg_id, "delta": leftover}
                        yield f"data: {json.dumps(chunk)}\n\n"
                    should_stop = True
                    break
            if should_stop:
                break
            chunk = {"id": msg_id, "delta": token_text}
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0)

        yield f"data: {json.dumps({'id': msg_id, 'delta': '', 'done': True})}\n\n"
        thread.join()

    except Exception as e:
        error_chunk = {"id": msg_id, "delta": f"\n\n⚠️ RAG streaming error: {str(e)}", "done": True}
        yield f"data: {json.dumps(error_chunk)}\n\n"


# ── Memory Management Endpoints ─────────────────────────────────────────────

@app.post("/api/memories/ingest")
async def ingest_memory(req: MemoryIngestRequest):
    """Manually ingest a memory into the RAG system."""
    if not rag_engine.initialized:
        raise HTTPException(503, "RAG engine not ready.")
    try:
        result = await rag_engine.ingest_memory(
            content=req.content, source=req.source, session_id=req.session_id
        )
        return {"status": "ok", "memory": result}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/memories")
async def get_memories(limit: int = Query(50, ge=1, le=500),
                       offset: int = Query(0, ge=0)):
    """Get stored memories with pagination."""
    if not rag_engine.initialized:
        raise HTTPException(503, "RAG engine not ready.")
    memories = rag_engine.get_memories(limit=limit, offset=offset)
    total = rag_engine.metadata_store.count_memories()
    return {"memories": memories, "total": total, "limit": limit, "offset": offset}


@app.post("/api/memories/search")
async def search_memories(req: MemorySearchRequest):
    """Search memories by semantic similarity."""
    if not rag_engine.initialized:
        raise HTTPException(503, "RAG engine not ready.")
    results = rag_engine.search_memories(query=req.query, top_k=req.top_k)
    return {"results": results, "count": len(results)}


@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a memory."""
    if not rag_engine.initialized:
        raise HTTPException(503, "RAG engine not ready.")
    success = rag_engine.delete_memory(memory_id)
    return {"status": "ok" if success else "not_found"}


# ── Knowledge Graph Endpoints ────────────────────────────────────────────────

@app.get("/api/graph")
async def get_graph():
    """Get knowledge graph data for visualization."""
    if not rag_engine.initialized:
        raise HTTPException(503, "RAG engine not ready.")
    return rag_engine.get_graph_data()


@app.get("/api/entities")
async def get_entities(limit: int = Query(100, ge=1, le=1000)):
    """Get all entities in the knowledge graph."""
    if not rag_engine.initialized:
        raise HTTPException(503, "RAG engine not ready.")
    return {"entities": rag_engine.get_entities(limit=limit)}


@app.get("/api/beliefs")
async def get_belief_deltas(limit: int = Query(50, ge=1, le=200)):
    """Get detected belief evolution events."""
    if not rag_engine.initialized:
        raise HTTPException(503, "RAG engine not ready.")
    return {"beliefs": rag_engine.get_belief_deltas(limit=limit)}


@app.get("/api/communities")
async def get_communities():
    """Get GraphRAG community summaries."""
    if not rag_engine.initialized:
        raise HTTPException(503, "RAG engine not ready.")
    return {"communities": rag_engine.get_community_summaries()}


# ── RAG System Stats ────────────────────────────────────────────────────────

@app.get("/api/rag/stats")
async def rag_stats():
    """Get comprehensive RAG system statistics."""
    return rag_engine.get_rag_stats()


@app.get("/api/rag/health")
async def rag_health():
    """RAG system health check."""
    return {
        "rag_initialized": rag_engine.initialized,
        "stats": rag_engine.get_rag_stats() if rag_engine.initialized else {},
    }


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=HOST, port=PORT, reload=False)
