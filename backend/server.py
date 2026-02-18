"""
FastAPI Backend Server for Cortex Lab - DeepSeek-R1-1.5B
Serves the model via REST API + Server-Sent Events (streaming)
"""

import os
import sys
import time
import json
import asyncio
import uuid
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from threading import Thread

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME = os.environ.get("MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
USE_4BIT   = os.environ.get("USE_4BIT", "false").lower() == "true"  # Disabled for 1.5B model
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
    print("  Cortex Lab  ·  DeepSeek-R1-1.5B  ·  FastAPI Backend")
    print("=" * 64 + "\n")

    # ── Tokenizer ────────────────────────────────────────────────────────
    print("[1/2] Loading tokenizer …")
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

    model_info = {
        "name": MODEL_NAME,
        "parameters": "14B",
        "quantization": quant,
        "device": gpu_name,
        "gpu_memory": gpu_mem,
        "max_context": 32768,
        "load_time_seconds": round(elapsed, 1),
    }

    print(f"  ✓ Model loaded in {elapsed:.1f}s  ({quant} on {gpu_name})")
    print(f"\n  Server ready → http://{HOST}:{PORT}\n")

    yield  # ← app runs here

    # cleanup
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="DeepSeek-R1 API",
    version="1.0.0",
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

# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_prompt(messages: list[ChatMessage]) -> str:
    """Build a plain prompt from the chat history."""
    parts: list[str] = []
    for m in messages:
        if m.role == "user":
            parts.append(f"User: {m.content}")
        elif m.role == "assistant":
            parts.append(f"Assistant: {m.content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def _split_thinking(text: str):
    """Separate <think>…</think> reasoning from visible answer."""
    thinking = None
    content  = text
    if "<think>" in text:
        start = text.index("<think>") + len("<think>")
        if "</think>" in text:
            end = text.index("</think>")
            thinking = text[start:end].strip()
            content  = text[end + len("</think>"):].strip()
        else:
            thinking = text[start:].strip()
            content  = ""
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
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=max(req.temperature, 0.01),
            top_p=req.top_p,
            do_sample=req.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
    thinking, content = _split_thinking(generated)

    return ChatResponse(
        id=f"msg-{uuid.uuid4().hex[:12]}",
        model=MODEL_NAME,
        created=int(time.time()),
        content=content or generated,
        thinking=thinking,
        usage={
            "prompt_tokens": input_len,
            "completion_tokens": out.shape[-1] - input_len,
            "total_tokens": out.shape[-1],
        },
    )


async def _stream_generate(prompt: str, req: ChatRequest):
    """Yield Server-Sent Events token by token."""
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = {
        **inputs,
        "max_new_tokens": req.max_tokens,
        "temperature": max(req.temperature, 0.01),
        "top_p": req.top_p,
        "do_sample": req.temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    msg_id = f"msg-{uuid.uuid4().hex[:12]}"

    for token_text in streamer:
        chunk = {"id": msg_id, "delta": token_text}
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0)  # yield control so FastAPI can flush

    yield f"data: {json.dumps({'id': msg_id, 'delta': '', 'done': True})}\n\n"
    thread.join()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=HOST, port=PORT, reload=False)
