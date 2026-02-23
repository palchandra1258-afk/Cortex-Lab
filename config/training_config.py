"""
Cortex Lab Training Configuration
DeepSeek-R1-Distill-Qwen-7B | RTX 4000 Ada Generation (20GB VRAM)
Ada Lovelace | Compute Capability 8.9 | BF16 Native

This is the single source of truth for all training hyperparameters.
Import from here in all training scripts.
"""

import torch
from pathlib import Path

# =============================================================================
# MODEL
# =============================================================================

# Use local model if downloaded, otherwise pull from HuggingFace
_LOCAL_7B = Path(__file__).resolve().parent.parent / "models" / "deepseek-r1-7b"
BASE_MODEL = str(_LOCAL_7B) if _LOCAL_7B.exists() and (_LOCAL_7B / "config.json").exists() \
             else "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

ULTRA_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"  # optional upgrade path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = str(_PROJECT_ROOT / "fine_tuned")
TRAINING_DATA_DIR = str(_PROJECT_ROOT / "training_data")

# =============================================================================
# 4-BIT QUANTIZATION (QLoRA — NF4 double quantization)
# =============================================================================

quantization_config = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": torch.bfloat16,   # Ada Lovelace native — no loss scaling issues
    "bnb_4bit_use_double_quant": True,           # Extra 0.4GB savings
    "bnb_4bit_quant_type": "nf4",                # Normal Float 4 — better than fp4 for LLMs
}

# =============================================================================
# LORA CONFIGURATIONS (per stage)
# ALL_MODULES targeting: q, k, v, o (attention) + gate, up, down (MLP)
# Research basis: Dettmers et al. 2024 — MLP targeting improves factual recall
# =============================================================================

# Target modules for 7B (Qwen2 architecture)
ATTN_MODULES     = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_MODULES      = ["gate_proj", "up_proj", "down_proj"]
ALL_MODULES      = ATTN_MODULES + MLP_MODULES
ATTN_ONLY        = ATTN_MODULES
ATTN_GATE        = ATTN_MODULES + ["gate_proj", "down_proj"]

LORA_CONFIGS = {

    # Stage 1: RAG-Grounded Faithfulness
    # High rank (64) — foundation stage, maximum capacity for behavior rewrite
    "stage1_faithfulness": {
        "r": 64,
        "lora_alpha": 128,
        "target_modules": ALL_MODULES,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },

    # Stage 2: Agentic Reasoning & Tool-Use
    # High rank — complex structured JSON output requires broad adaptation
    "stage2_agentic": {
        "r": 64,
        "lora_alpha": 128,
        "target_modules": ALL_MODULES,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },

    # Stage 3: Causal Chain & Temporal Reasoning
    # Medium rank — builds on Stage 1 grounding, targeted new learning
    "stage3_causal": {
        "r": 32,
        "lora_alpha": 64,
        "target_modules": ALL_MODULES,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },

    # Stage 4: Self-Reflective Critique (Self-RAG / CRAG)
    # High rank — entirely new token patterns (ISREL/ISSUP/ISUSE)
    "stage4_selfrag": {
        "r": 64,
        "lora_alpha": 128,
        "target_modules": ALL_MODULES,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },

    # Stage 5: Belief Evolution & Contradiction Handling
    # Medium rank — specialized temporal comparison task
    "stage5_belief": {
        "r": 32,
        "lora_alpha": 64,
        "target_modules": ALL_MODULES,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },

    # Stage 6: Memory Consolidation & Summarization
    # Medium rank — compression reasoning, targeted MLP modules
    "stage6_summarization": {
        "r": 32,
        "lora_alpha": 64,
        "target_modules": ATTN_GATE,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },

    # Stage 7: Multi-Turn Dialogue Coherence
    # High-medium rank — context tracking across turns needs broad capacity
    "stage7_dialogue": {
        "r": 48,
        "lora_alpha": 96,
        "target_modules": ALL_MODULES,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },

    # Stage 8: Long-Context Multi-Hop Reasoning
    # High rank — deep reasoning chains across 10-20 memory chunks
    "stage8_longcontext": {
        "r": 64,
        "lora_alpha": 128,
        "target_modules": ALL_MODULES,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },

    # Stage 9: DPO Preference Alignment
    # Medium rank, attention-focused — preference tuning, not skill learning
    "stage9_dpo": {
        "r": 32,
        "lora_alpha": 64,
        "target_modules": ATTN_ONLY,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },

    # Stage 10: User Style Adaptation (NOT merged — stays as hot-swap adapter)
    # Minimal rank — lightweight personalization, minimal footprint
    "stage10_user_style": {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],   # Attention only for style
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },

    # ─── EXTENDED PIPELINE (Stages 11-15) ─────────────────────────────────
    # These stages run AFTER stages 1-10 complete.
    # They build on the stage-10 merged checkpoint.

    # Stage 11: ORPO — Odds-Ratio Preference Optimization
    # Replaces DPO with a simpler, reference-free preference alignment
    # Medium rank, attention-only — preference alignment, not skill learning
    "stage11_orpo": {
        "r": 32,
        "lora_alpha": 64,
        "target_modules": ATTN_ONLY,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },

    # Stage 12: RAFT — Retrieval-Augmented Fine Tuning
    # Teaches the model to reason over noisy retrieval context (distractor docs)
    # High rank — complex input filtering + reasoning + citation
    "stage12_raft": {
        "r": 64,
        "lora_alpha": 128,
        "target_modules": ALL_MODULES,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },

    # Stage 13: Function-Calling Fine-Tuning
    # Trains structured JSON tool-call output with strict schema adherence
    # High rank — structured output needs broad adaptation
    "stage13_function_calling": {
        "r": 64,
        "lora_alpha": 128,
        "target_modules": ALL_MODULES,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },

    # Stage 14: Rejection Sampling Fine-Tuning (RFT)
    # SFT on best-of-N filtered examples — only the highest quality outputs
    # Medium rank — refinement stage, not new skill learning
    "stage14_rft": {
        "r": 32,
        "lora_alpha": 64,
        "target_modules": ALL_MODULES,
        "lora_dropout": 0.03,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },

    # Stage 15: SPIN — Self-Play Improvement
    # Iterative self-play: model learns to distinguish its own outputs
    # from ground truth, closing the gap each round
    # Medium rank, attention-focused — subtle distribution alignment
    "stage15_spin": {
        "r": 32,
        "lora_alpha": 64,
        "target_modules": ATTN_ONLY,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
}

# =============================================================================
# TRAINING CONFIGURATIONS (per stage)
# RTX 4000 Ada Generation — No gradient checkpointing (have VRAM headroom)
# bf16 native, full-precision AdamW, torch.compile, batch=4, seq=2048
# =============================================================================

TRAINING_CONFIGS = {

    "stage1_faithfulness": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,          # Effective batch = 16
        "gradient_checkpointing": True,            # Enabled — cuts activation VRAM ~60%
        "learning_rate": 2e-4,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch",                    # Full-precision, better convergence
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "bf16": True,                              # Ada Lovelace native
        "torch_compile": False,                    # Disabled — incompatible with grad checkpointing + bnb
        "group_by_length": True,
        "dataloader_num_workers": 2,
        "max_seq_length": 1024,
    },

    "stage2_agentic": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch",
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "bf16": True,
        "torch_compile": False,
        "group_by_length": True,
        "dataloader_num_workers": 2,
        "max_seq_length": 1024,
    },

    "stage3_causal": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "learning_rate": 1.5e-4,                   # Slightly lower — building on stage 1
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch",
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "bf16": True,
        "torch_compile": False,
        "group_by_length": True,
        "dataloader_num_workers": 2,
        "max_seq_length": 1024,
    },

    "stage4_selfrag": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.05,                      # Longer warmup for new token patterns
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch",
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "bf16": True,
        "torch_compile": False,
        "group_by_length": True,
        "dataloader_num_workers": 2,
        "max_seq_length": 1024,
    },

    "stage5_belief": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "learning_rate": 1.5e-4,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch",
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "bf16": True,
        "torch_compile": False,
        "group_by_length": True,
        "dataloader_num_workers": 2,
        "max_seq_length": 1024,
    },

    "stage6_summarization": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "learning_rate": 1.5e-4,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch",
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "bf16": True,
        "torch_compile": False,
        "group_by_length": True,
        "dataloader_num_workers": 2,
        "max_seq_length": 1024,
    },

    "stage7_dialogue": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "learning_rate": 1.5e-4,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch",
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "bf16": True,
        "torch_compile": False,
        "group_by_length": True,
        "dataloader_num_workers": 2,
        "max_seq_length": 1024,
    },

    "stage8_longcontext": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,          # Smallest batch — very long seqs
        "gradient_accumulation_steps": 16,         # Maintain effective batch = 16
        "gradient_checkpointing": True,
        "learning_rate": 1e-4,                     # Conservative — most complex reasoning
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch",
        "weight_decay": 0.01,
        "max_grad_norm": 0.5,                      # Tighter clipping for stability
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "bf16": True,
        "torch_compile": False,
        "group_by_length": True,
        "dataloader_num_workers": 2,
        "max_seq_length": 2048,                    # Long context needs 2048
    },

    "stage9_dpo": {
        # DPO uses DPOConfig, not TrainingArguments — these map to DPOConfig fields
        "learning_rate": 5e-6,                     # Very conservative for alignment
        "num_train_epochs": 1,                     # Single pass for DPO
        "per_device_train_batch_size": 1,          # DPO needs 2x memory (chosen + rejected)
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "warmup_ratio": 0.1,
        "beta": 0.1,                               # KL divergence penalty
        "max_length": 1024,
        "max_prompt_length": 512,
        "bf16": True,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "max_seq_length": 1024,
    },

    "stage10_user_style": {
        "num_train_epochs": 2,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "gradient_checkpointing": True,
        "learning_rate": 5e-5,                     # Lower LR — don't overwrite previous stages
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch",
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "logging_steps": 5,
        "save_strategy": "epoch",
        "save_total_limit": 3,
        "bf16": True,
        "torch_compile": False,
        "group_by_length": True,
        "dataloader_num_workers": 2,
        "max_seq_length": 1024,                    # Shorter for style examples
    },

    # ─── EXTENDED PIPELINE (Stages 11-15) ─────────────────────────────────

    "stage11_orpo": {
        # ORPO uses ORPOConfig — reference-free preference optimization
        # Key advantage over DPO: no reference model needed → half the memory
        "learning_rate": 8e-6,                     # Higher than DPO — ORPO has built-in regularization
        "num_train_epochs": 1,                     # Single pass for preference alignment
        "per_device_train_batch_size": 1,          # ORPO needs 2x memory (chosen + rejected)
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "warmup_ratio": 0.1,
        "beta": 0.1,                               # ORPO odds ratio weight
        "max_length": 1024,
        "bf16": True,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "max_seq_length": 1024,
        "optim": "adamw_torch",
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
    },

    "stage12_raft": {
        # RAFT: Retrieval-Augmented Fine Tuning
        # Model sees oracle doc + N distractor docs and must identify & use the right one
        "num_train_epochs": 2,
        "per_device_train_batch_size": 1,          # Longer inputs: query + 5 docs
        "gradient_accumulation_steps": 16,         # Effective batch = 16
        "gradient_checkpointing": True,
        "learning_rate": 1e-4,                     # Moderate — new skill
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch",
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "bf16": True,
        "torch_compile": False,
        "group_by_length": True,
        "dataloader_num_workers": 2,
        "max_seq_length": 2048,                    # Multi-doc context needs 2048
    },

    "stage13_function_calling": {
        # Function-Calling FT: strict structured JSON output
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "learning_rate": 1.5e-4,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch",
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "bf16": True,
        "torch_compile": False,
        "group_by_length": True,
        "dataloader_num_workers": 2,
        "max_seq_length": 1024,
    },

    "stage14_rft": {
        # Rejection Sampling FT: train only on best-of-N outputs
        # Conservative LR — this is a refinement stage, not skill learning
        "num_train_epochs": 2,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "learning_rate": 5e-5,                     # Very conservative — polishing
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch",
        "weight_decay": 0.01,
        "max_grad_norm": 0.5,                      # Tight clipping for stability
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "bf16": True,
        "torch_compile": False,
        "group_by_length": True,
        "dataloader_num_workers": 2,
        "max_seq_length": 1024,
    },

    "stage15_spin": {
        # SPIN Self-Play: DPO-like training where rejected = model's own output
        # Uses DPO-style config with prompt/chosen/rejected
        "learning_rate": 5e-6,                     # Very conservative — subtle alignment
        "num_train_epochs": 1,                     # Single pass per SPIN iteration
        "per_device_train_batch_size": 1,          # Preference pairs need 2x memory
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "warmup_ratio": 0.1,
        "beta": 0.1,                               # KL divergence penalty (DPO-style)
        "max_length": 1024,
        "max_prompt_length": 512,
        "bf16": True,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "max_seq_length": 1024,
        "optim": "adamw_torch",
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
    },
}

# =============================================================================
# VRAM BUDGET REFERENCE (RTX 4000 Ada Generation — 20,480 MB)
# =============================================================================
#
# TRAINING (7B QLoRA r=64, batch=4, seq=2048):
#   Base model (4-bit NF4 + double quant): ~4,200 MB
#   LoRA weights (r=64, ALL_MODULES):        ~460 MB
#   Full-precision AdamW optimizer:        ~1,800 MB
#   Activations (no grad checkpoint):      ~3,200 MB
#   Forward pass working memory:             ~800 MB
#   KV cache (batch=4, seq=2048):          ~1,200 MB
#   CUDA + torch.compile overhead:           ~600 MB
#   bf16 compute buffers:                    ~740 MB
#   TOTAL:                               ~13,000 MB  (63% utilization ✅)
#   HEADROOM:                             ~7,480 MB  (safety buffer)
#
# INFERENCE (7B merged, seq=4096):
#   Model (4-bit):                         ~4,200 MB
#   User LoRA (Stage 10, r=16):               ~35 MB
#   KV cache (seq=4096):                     ~600 MB
#   BGE-large-en-v1.5:                     ~1,340 MB
#   BGE-reranker-v2-m3:                      ~560 MB
#   CUDA overhead:                           ~300 MB
#   TOTAL:                                ~7,035 MB  (34% utilization ✅)
#   HEADROOM:                            ~13,445 MB  (for batching/long ctx)
#
# 14B ULTRA TRAINING (optional):
#   Base model (4-bit):                    ~8,400 MB
#   LoRA (r=32, attn-only):                  ~280 MB
#   AdamW 8-bit:                           ~1,200 MB
#   Activations (with grad checkpoint):    ~2,800 MB
#   KV cache (batch=2):                      ~600 MB
#   CUDA overhead:                           ~500 MB
#   TOTAL:                               ~13,780 MB  (67% utilization ✅)
#   HEADROOM:                             ~6,700 MB  (tight but fits)
