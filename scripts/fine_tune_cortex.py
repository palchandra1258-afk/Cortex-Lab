#!/usr/bin/env python3
"""
Cortex Lab — Fine-Tuning Pipeline
==================================
CortexLabTrainer: Sequential 15-stage curriculum fine-tuning of
DeepSeek-R1-Distill-Qwen-7B on RTX 4000 Ada Generation (20GB VRAM).

Architecture:
  • Stages 1–8, 10, 12–14  → SFTTrainer (trl)
  • Stage 9                 → DPOTrainer (trl)
  • Stage 11                → ORPOTrainer (trl)
  • Stage 15                → DPOTrainer (SPIN self-play)
  • Each stage loads the previous stage's merged adapter as its base
  • Stage 10 is NEVER merged (hot-swap LoRA per user)

Usage:
  python scripts/fine_tune_cortex.py --stage stage1_faithfulness
  python scripts/fine_tune_cortex.py --all
  python scripts/fine_tune_cortex.py --all --resume
  python scripts/fine_tune_cortex.py --extended           # Stages 11-15 only
  python scripts/fine_tune_cortex.py --extended --resume
  python scripts/fine_tune_cortex.py --status

Hardware target: RTX 4000 Ada Gen | 20,480 MB VRAM | BF16 | CC 8.9
"""

import os
import sys
import json
import time
import shutil
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

# Prevent CUDA memory fragmentation OOM between stages
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ─── Add project root to sys.path ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CortexLab")

# ─── Lazy imports (heavy — only after CLI parse) ─────────────────────────────
def _import_training_deps():
    """Import all heavy training dependencies."""
    global torch, transformers, peft, trl, datasets
    global AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    global LoraConfig, get_peft_model, PeftModel
    global SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
    global ORPOTrainer, ORPOConfig
    global Dataset, load_dataset

    import torch
    import transformers
    import peft
    import trl
    import datasets

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, PeftModel
    from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
    from trl import ORPOTrainer, ORPOConfig
    from datasets import Dataset, load_dataset

    logger.info(f"torch        {torch.__version__}")
    logger.info(f"transformers {transformers.__version__}")
    logger.info(f"peft         {peft.__version__}")
    logger.info(f"trl          {trl.__version__}")
    logger.info(f"datasets     {datasets.__version__}")


# ─── Config ──────────────────────────────────────────────────────────────────
from config.training_config import (
    BASE_MODEL,
    OUTPUT_DIR,
    TRAINING_DATA_DIR,
    quantization_config,
    LORA_CONFIGS,
    TRAINING_CONFIGS,
)

STAGE_ORDER = [
    "stage1_faithfulness",
    "stage2_agentic",
    "stage3_causal",
    "stage4_selfrag",
    "stage5_belief",
    "stage6_summarization",
    "stage7_dialogue",
    "stage8_longcontext",
    "stage9_dpo",
    "stage10_user_style",
]

# Extended stages (run after base 10 complete)
EXTENDED_STAGE_ORDER = [
    "stage11_orpo",
    "stage12_raft",
    "stage13_function_calling",
    "stage14_rft",
    "stage15_spin",
]

# Full 15-stage order
FULL_STAGE_ORDER = STAGE_ORDER + EXTENDED_STAGE_ORDER

# Stage 9 uses DPO, Stage 15 (SPIN) uses DPO-style training
DPO_STAGES = {"stage9_dpo", "stage15_spin"}

# Stage 11 uses ORPO (reference-free preference optimization)
ORPO_STAGES = {"stage11_orpo"}

# Stage 10 is NEVER merged — stays as hot-swap LoRA
NEVER_MERGE = {"stage10_user_style"}

# SFT stages = everything that's not DPO or ORPO
SFT_STAGES = set(FULL_STAGE_ORDER) - DPO_STAGES - ORPO_STAGES

STAGE_DATA_FILES = {
    "stage1_faithfulness":    "stage1_faithfulness.json",
    "stage2_agentic":         "stage2_agentic.json",
    "stage3_causal":          "stage3_causal.json",
    "stage4_selfrag":         "stage4_selfrag.json",
    "stage5_belief":          "stage5_belief.json",
    "stage6_summarization":   "stage6_summarization.json",
    "stage7_dialogue":        "stage7_dialogue.json",
    "stage8_longcontext":     "stage8_longcontext.json",
    "stage9_dpo":             "stage9_dpo.json",
    "stage10_user_style":     "stage10_user_style.json",
    # Extended stages
    "stage11_orpo":             "stage11_orpo.json",
    "stage12_raft":             "stage12_raft.json",
    "stage13_function_calling": "stage13_function_calling.json",
    "stage14_rft":              "stage14_rft.json",
    "stage15_spin":             "stage15_spin.json",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_sft(ex: dict) -> str:
    """Format a SFTExample dict into a training string."""
    instr = ex.get("instruction", "").strip()
    inp   = ex.get("input", "").strip()
    out   = ex.get("output", "").strip()

    if inp:
        text = (
            f"<|im_start|>system\n{instr}<|im_end|>\n"
            f"<|im_start|>user\n{inp}<|im_end|>\n"
            f"<|im_start|>assistant\n{out}<|im_end|>"
        )
    else:
        text = (
            f"<|im_start|>system\nYou are Cortex, an AI with persistent memory.<|im_end|>\n"
            f"<|im_start|>user\n{instr}<|im_end|>\n"
            f"<|im_start|>assistant\n{out}<|im_end|>"
        )
    return text


def load_sft_dataset(stage: str) -> "Dataset":
    """Load and format SFT dataset for a given stage."""
    data_path = Path(TRAINING_DATA_DIR) / STAGE_DATA_FILES[stage]
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            f"Run: python scripts/generate_datasets.py --stage {stage}"
        )

    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    logger.info(f"Loaded {len(raw)} raw examples from {data_path.name}")

    formatted = []
    for ex in raw:
        try:
            text = _fmt_sft(ex)
            if len(text) >= 100:
                formatted.append({"text": text})
        except Exception:
            continue

    logger.info(f"Formatted {len(formatted)} examples for training")
    return Dataset.from_list(formatted)


def load_dpo_dataset(stage: str) -> "Dataset":
    """Load and format DPO dataset for stage 9."""
    data_path = Path(TRAINING_DATA_DIR) / STAGE_DATA_FILES[stage]
    if not data_path.exists():
        raise FileNotFoundError(f"DPO dataset not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    logger.info(f"Loaded {len(raw)} raw DPO examples from {data_path.name}")

    formatted = []
    for ex in raw:
        try:
            prompt   = ex.get("prompt", "").strip()
            chosen   = ex.get("chosen", "").strip()
            rejected = ex.get("rejected", "").strip()
            if not (prompt and chosen and rejected):
                continue
            if chosen == rejected:
                continue
            formatted.append({
                "prompt":   f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                "chosen":   chosen + "<|im_end|>",
                "rejected": rejected + "<|im_end|>",
            })
        except Exception:
            continue

    logger.info(f"Formatted {len(formatted)} DPO pairs for training")
    return Dataset.from_list(formatted)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def get_base_model_path(stage: str) -> str:
    """
    Determine the base model path for a given stage.
    Each stage builds on the previous stage's merged adapter.
    Stage 1 always starts from the base HuggingFace model.
    
    Special cases:
    - Stage 10 (user_style) is NEVER merged, so stage 11+ uses stage 9's merged model
    - If a previous stage's merged model was cleaned up, we re-merge from its adapter
    """
    idx = FULL_STAGE_ORDER.index(stage)
    if idx == 0:
        return BASE_MODEL

    # Walk backwards to find the most recent usable merged model
    # Skip NEVER_MERGE stages (they don't produce a merged model)
    for prev_idx in range(idx - 1, -1, -1):
        prev_stage = FULL_STAGE_ORDER[prev_idx]
        
        # Skip stages that are never merged (e.g., stage10_user_style)
        if prev_stage in NEVER_MERGE:
            continue
            
        merged_path = Path(OUTPUT_DIR) / prev_stage / "merged"
        if merged_path.exists() and (merged_path / "config.json").exists():
            logger.info(f"Loading merged model from stage: {prev_stage}")
            return str(merged_path)

        # Check for adapter — if it exists but merged was cleaned, re-merge it first
        adapter_path = Path(OUTPUT_DIR) / prev_stage / "adapter"
        if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
            logger.info(
                f"Previous stage '{prev_stage}' has adapter but no merged model. "
                f"Re-merging from base model..."
            )
            # Re-merge: load base model + apply adapter
            _remerge_adapter(prev_stage)
            merged_path = Path(OUTPUT_DIR) / prev_stage / "merged"
            if merged_path.exists():
                return str(merged_path)

        # Check if stage was completed + fully cleaned (no adapter or merged)
        meta_path = Path(OUTPUT_DIR) / prev_stage / "training_meta.json"
        if meta_path.exists():
            # Stage completed but everything cleaned — keep walking back
            continue

    logger.warning(
        f"No merged checkpoint found in any previous stage for '{stage}'. "
        f"Falling back to base model. Run stages in order for best results."
    )
    return BASE_MODEL


def _remerge_adapter(stage: str):
    """
    Re-merge an adapter whose merged model was cleaned up.
    Walks back to find the nearest available base, loads it on CPU,
    applies the adapter, and saves the merged model.
    """
    from peft import PeftModel

    adapter_dir = Path(OUTPUT_DIR) / stage / "adapter"
    merged_dir = Path(OUTPUT_DIR) / stage / "merged"

    # Find what base model this adapter needs
    # Walk backwards from this stage to find a usable base
    idx = FULL_STAGE_ORDER.index(stage)
    base_path = BASE_MODEL
    for prev_idx in range(idx - 1, -1, -1):
        prev_stage = FULL_STAGE_ORDER[prev_idx]
        if prev_stage in NEVER_MERGE:
            continue
        prev_merged = Path(OUTPUT_DIR) / prev_stage / "merged"
        if prev_merged.exists() and (prev_merged / "config.json").exists():
            base_path = str(prev_merged)
            break

    logger.info(f"Re-merging {stage} adapter. Base: {base_path}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    peft_model = PeftModel.from_pretrained(
        base_model, str(adapter_dir),
        is_trainable=False,
    )
    merged = peft_model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir))

    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    tokenizer.save_pretrained(str(merged_dir))

    logger.info(f"Re-merged model saved → {merged_dir}")

    del merged, peft_model, base_model
    import gc; gc.collect()


def load_model_and_tokenizer(
    model_path: str,
    stage: str,
    is_dpo: bool = False,
    is_orpo: bool = False,
) -> tuple:
    """
    Load base model (4-bit QLoRA) + tokenizer, attach LoRA adapter.
    For DPO/ORPO, returns model WITHOUT LoRA (trainer applies it).
    """
    from transformers import BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, PeftModel

    # ── Quantization config ─────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quantization_config["load_in_4bit"],
        bnb_4bit_compute_dtype=quantization_config["bnb_4bit_compute_dtype"],
        bnb_4bit_use_double_quant=quantization_config["bnb_4bit_use_double_quant"],
        bnb_4bit_quant_type=quantization_config["bnb_4bit_quant_type"],
    )

    logger.info(f"Loading base model: {model_path}")
    logger.info(f"Quantization: 4-bit NF4 + double quant | compute: bf16")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",      # Avoid flash-attn dependency
    )

    # ── Tokenizer ───────────────────────────────────────────────────────────
    tokenizer_path = BASE_MODEL  # Always load tokenizer from HF base (consistent vocab)
    if Path(model_path).exists() and (Path(model_path) / "tokenizer.json").exists():
        tokenizer_path = model_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── LoRA config for this stage ──────────────────────────────────────────
    lora_cfg = LORA_CONFIGS[stage]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )

    if not is_dpo and not is_orpo:
        model = get_peft_model(model, lora_config)
        # Required for gradient checkpointing with 4-bit quantized models
        model.enable_input_require_grads()
        model.print_trainable_parameters()

    return model, tokenizer, lora_config


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_sft_stage(
    stage: str,
    model,
    tokenizer,
    dataset: "Dataset",
    output_dir: Path,
    cfg: dict,
    resume_from_checkpoint: Optional[str] = None,
):
    """Train a single SFT stage using trl.SFTTrainer."""

    sft_config = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg.get("gradient_checkpointing") else None,
        learning_rate=cfg["learning_rate"],
        warmup_steps=int(cfg.get("warmup_ratio", 0.05) * cfg["num_train_epochs"] * 100),
        lr_scheduler_type=cfg["lr_scheduler_type"],
        optim=cfg["optim"],
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        save_total_limit=cfg["save_total_limit"],
        bf16=cfg["bf16"],
        dataloader_num_workers=cfg.get("dataloader_num_workers", 2),
        max_length=cfg.get("max_seq_length", 1024),
        dataset_text_field="text",
        report_to="none",               # No wandb/tensorboard dependency
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info(f"Starting SFT training: {stage}")
    logger.info(f"  Examples: {len(dataset)}")
    logger.info(f"  Epochs:   {cfg['num_train_epochs']}")
    logger.info(f"  Batch:    {cfg['per_device_train_batch_size']} × {cfg['gradient_accumulation_steps']} = {cfg['per_device_train_batch_size'] * cfg['gradient_accumulation_steps']} effective")
    logger.info(f"  LR:       {cfg['learning_rate']}")
    logger.info(f"  Seq len:  {cfg.get('max_seq_length', 2048)}")

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    elapsed = time.time() - t0

    logger.info(f"Training complete in {timedelta(seconds=int(elapsed))}")
    return trainer


def train_dpo_stage(
    stage: str,
    model,
    tokenizer,
    lora_config: "LoraConfig",
    dataset: "Dataset",
    output_dir: Path,
    cfg: dict,
    resume_from_checkpoint: Optional[str] = None,
):
    """Train Stage 9 using trl.DPOTrainer."""
    from peft import get_peft_model

    dpo_config = DPOConfig(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        beta=cfg.get("beta", 0.1),
        max_length=cfg.get("max_length", 2048),
        max_prompt_length=cfg.get("max_prompt_length", 1024),
        bf16=cfg["bf16"],
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        save_total_limit=cfg["save_total_limit"],
        report_to="none",
        remove_unused_columns=False,
    )

    # Apply LoRA for DPO
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainer = DPOTrainer(
        model=model,
        ref_model=None,         # Use implicit reference (PEFT frozen base)
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info(f"Starting DPO training: {stage}")
    logger.info(f"  Examples: {len(dataset)}")
    logger.info(f"  Beta:     {cfg.get('beta', 0.1)}")
    logger.info(f"  LR:       {cfg['learning_rate']}")

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    elapsed = time.time() - t0

    logger.info(f"DPO training complete in {timedelta(seconds=int(elapsed))}")
    return trainer


def train_orpo_stage(
    stage: str,
    model,
    tokenizer,
    lora_config: "LoraConfig",
    dataset: "Dataset",
    output_dir: Path,
    cfg: dict,
    resume_from_checkpoint: Optional[str] = None,
):
    """Train using trl.ORPOTrainer — reference-free preference optimization."""
    from peft import get_peft_model

    orpo_config = ORPOConfig(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg.get("gradient_checkpointing") else None,
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        beta=cfg.get("beta", 0.1),
        max_length=cfg.get("max_length", 1024),
        bf16=cfg["bf16"],
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        save_total_limit=cfg["save_total_limit"],
        optim=cfg.get("optim", "adamw_torch"),
        weight_decay=cfg.get("weight_decay", 0.01),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        report_to="none",
        remove_unused_columns=False,
    )

    # Apply LoRA for ORPO
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    trainer = ORPOTrainer(
        model=model,
        args=orpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info(f"Starting ORPO training: {stage}")
    logger.info(f"  Examples: {len(dataset)}")
    logger.info(f"  Beta:     {cfg.get('beta', 0.1)}")
    logger.info(f"  LR:       {cfg['learning_rate']}")
    logger.info(f"  NOTE:     No reference model needed (ORPO is reference-free)")

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    elapsed = time.time() - t0

    logger.info(f"ORPO training complete in {timedelta(seconds=int(elapsed))}")
    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTER SAVE & MERGE
# ─────────────────────────────────────────────────────────────────────────────

def save_adapter(trainer, output_dir: Path, stage: str):
    """Save the LoRA adapter weights."""
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    trainer.processing_class.save_pretrained(str(adapter_dir))
    logger.info(f"Adapter saved → {adapter_dir}")


def merge_adapter(output_dir: Path, stage: str):
    """
    Merge LoRA adapter into base model weights and save as merged model.
    Skipped for Stage 10 (hot-swap LoRA — never merged).
    """
    if stage in NEVER_MERGE:
        logger.info(f"Stage {stage} is a hot-swap adapter — skipping merge.")
        return

    adapter_dir = output_dir / "adapter"
    merged_dir  = output_dir / "merged"

    if not adapter_dir.exists():
        logger.error(f"Adapter not found: {adapter_dir}")
        return

    logger.info(f"Merging adapter → {merged_dir}")

    from peft import PeftModel
    from transformers import BitsAndBytesConfig

    # Reload base model in full precision for merge
    # Determine what the base was for this stage
    idx = FULL_STAGE_ORDER.index(stage)
    if idx == 0:
        base_path = BASE_MODEL
    else:
        # Walk backward to find the most recent merged model (skip NEVER_MERGE)
        base_path = BASE_MODEL
        for prev_idx in range(idx - 1, -1, -1):
            prev_stage = FULL_STAGE_ORDER[prev_idx]
            if prev_stage in NEVER_MERGE:
                continue
            prev_merged = Path(OUTPUT_DIR) / prev_stage / "merged"
            if prev_merged.exists() and (prev_merged / "config.json").exists():
                base_path = str(prev_merged)
                break
            # If adapter exists but merged was cleaned, re-merge it
            prev_adapter = Path(OUTPUT_DIR) / prev_stage / "adapter"
            if prev_adapter.exists() and (prev_adapter / "adapter_config.json").exists():
                logger.info(f"Re-merging {prev_stage} adapter before merge...")
                _remerge_adapter(prev_stage)
                if prev_merged.exists():
                    base_path = str(prev_merged)
                    break

    logger.info(f"Loading base for merge: {base_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",           # CPU merge — avoids VRAM spike
        trust_remote_code=True,
    )

    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged = peft_model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir))

    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    tokenizer.save_pretrained(str(merged_dir))

    logger.info(f"Merged model saved → {merged_dir}")

    # Explicit cleanup to free CPU RAM after merge
    del merged, peft_model, base_model
    import gc; gc.collect()

    # Save merge metadata
    meta = {
        "stage": stage,
        "merged_at": datetime.now().isoformat(),
        "base_model": base_path,
        "adapter": str(adapter_dir),
    }
    (merged_dir / "merge_meta.json").write_text(json.dumps(meta, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# STATUS TRACKING
# ─────────────────────────────────────────────────────────────────────────────

def get_stage_status(stage: str) -> str:
    """Return status of a training stage."""
    out = Path(OUTPUT_DIR) / stage
    merged = out / "merged"
    adapter = out / "adapter"
    checkpoints = out / "checkpoints"
    meta = out / "training_meta.json"

    # training_meta.json is written on completion — definitive proof of done
    if meta.exists():
        if merged.exists():
            return "✅ merged"
        if adapter.exists():
            return "✅ adapter (not merged)"
        # Completed but merged/adapter cleaned up for disk space
        return "✅ complete (cleaned)"
    if merged.exists():
        return "✅ merged"
    if adapter.exists():
        return "⚡ adapter (not merged)"
    if checkpoints.exists() and any(checkpoints.iterdir()):
        return "🔄 in-progress"
    return "⬜ not started"


def print_status():
    """Print training status for all stages."""
    print("\n" + "=" * 70)
    print(f"  CORTEX LAB — Training Status (15-Stage Pipeline)")
    print(f"  Base: {BASE_MODEL}")
    print("=" * 70)

    total_done = 0
    for label, stages in [("BASE (1-10)", STAGE_ORDER), ("EXTENDED (11-15)", EXTENDED_STAGE_ORDER)]:
        print(f"\n  ─── {label} ───")
        for stage in stages:
            status = get_stage_status(stage)
            data_file = Path(TRAINING_DATA_DIR) / STAGE_DATA_FILES.get(stage, "")
            data_info = ""
            if data_file.exists():
                size_mb = data_file.stat().st_size / 1_048_576
                with open(data_file, "r") as f:
                    count = len(json.load(f))
                data_info = f"{count:,} examples ({size_mb:.1f} MB)"
            else:
                data_info = "⚠️  dataset missing"

            done = "✅" in status or "⚡" in status
            if done:
                total_done += 1

            trainer_label = ""
            if stage in DPO_STAGES:
                trainer_label = " [DPO]"
            elif stage in ORPO_STAGES:
                trainer_label = " [ORPO]"
            elif stage in NEVER_MERGE:
                trainer_label = " [hot-swap]"

            print(f"  {stage:<30} {status:<28} {data_info}{trainer_label}")

    print("\n" + "=" * 70)
    print(f"  Completed: {total_done}/{len(FULL_STAGE_ORDER)} stages")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# VRAM CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_vram():
    """Check VRAM availability before training."""
    if not torch.cuda.is_available():
        logger.error("CUDA not available. Training requires a GPU.")
        sys.exit(1)

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    total_mb = props.total_memory / 1_048_576
    free_mb = (props.total_memory - torch.cuda.memory_allocated(device)) / 1_048_576

    logger.info(f"GPU: {props.name}")
    logger.info(f"VRAM: {total_mb:.0f} MB total | {free_mb:.0f} MB free")
    logger.info(f"Compute Capability: {props.major}.{props.minor}")

    if total_mb < 15_000:
        logger.warning(f"Less than 15GB VRAM detected ({total_mb:.0f} MB). Training may OOM.")
        logger.warning("Consider reducing per_device_train_batch_size in config/training_config.py")

    if props.major < 8:
        logger.warning("BF16 requires compute capability >= 8.0. Falling back to FP16.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class CortexLabTrainer:
    """
    Orchestrates 15-stage sequential curriculum fine-tuning.

    Each stage:
    1. Determines the correct base model (HF or previous merged checkpoint)
    2. Loads model in 4-bit QLoRA
    3. Attaches stage-specific LoRA adapter
    4. Trains with SFTTrainer, DPOTrainer, or ORPOTrainer
    5. Saves adapter
    6. Merges adapter into weights (except Stage 10)
    7. Records completion metadata
    8. Cleans up previous stage artifacts to save disk
    """

    def __init__(self, resume: bool = False):
        self.resume = resume
        self.output_root = Path(OUTPUT_DIR)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def _stage_output_dir(self, stage: str) -> Path:
        d = self.output_root / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _is_complete(self, stage: str) -> bool:
        status = get_stage_status(stage)
        return "✅" in status or ("stage10" in stage and "⚡" in status)

    def _find_resume_checkpoint(self, stage: str) -> Optional[str]:
        """Find the latest checkpoint to resume from."""
        ckpt_dir = self._stage_output_dir(stage) / "checkpoints"
        if not ckpt_dir.exists():
            return None
        checkpoints = sorted(
            [d for d in ckpt_dir.iterdir() if d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[-1]),
        )
        if checkpoints:
            logger.info(f"Resuming from checkpoint: {checkpoints[-1]}")
            return str(checkpoints[-1])
        return None

    def run_stage(self, stage: str):
        """Run a single training stage (SFT, DPO, or ORPO)."""
        if stage not in FULL_STAGE_ORDER:
            raise ValueError(f"Unknown stage: {stage}. Valid: {FULL_STAGE_ORDER}")

        output_dir = self._stage_output_dir(stage)
        is_dpo = stage in DPO_STAGES
        is_orpo = stage in ORPO_STAGES

        # ── Skip if already complete (unless --force) ─────────────────────
        if self._is_complete(stage):
            has_checkpoint = self._find_resume_checkpoint(stage) is not None
            if not (self.resume and has_checkpoint):
                logger.info(f"Stage {stage} already complete. Skipping.")
                return

        # ── Resume checkpoint ─────────────────────────────────────────────
        resume_from = None
        if self.resume:
            resume_from = self._find_resume_checkpoint(stage)

        # ── Load dataset ──────────────────────────────────────────────────
        logger.info(f"\n{'='*60}")
        logger.info(f"STAGE: {stage.upper()}")
        logger.info(f"{'='*60}")

        if is_dpo or is_orpo:
            dataset = load_dpo_dataset(stage)
        else:
            dataset = load_sft_dataset(stage)

        # ── Determine base model ──────────────────────────────────────────
        base_path = get_base_model_path(stage)

        # ── Load model + tokenizer ────────────────────────────────────────
        model, tokenizer, lora_config = load_model_and_tokenizer(
            base_path, stage, is_dpo=is_dpo, is_orpo=is_orpo
        )

        # ── Get training config ───────────────────────────────────────────
        cfg = TRAINING_CONFIGS[stage]

        # ── Train ─────────────────────────────────────────────────────────
        if is_orpo:
            trainer = train_orpo_stage(
                stage, model, tokenizer, lora_config,
                dataset, output_dir, cfg, resume_from
            )
        elif is_dpo:
            trainer = train_dpo_stage(
                stage, model, tokenizer, lora_config,
                dataset, output_dir, cfg, resume_from
            )
        else:
            trainer = train_sft_stage(
                stage, model, tokenizer,
                dataset, output_dir, cfg, resume_from
            )

        # ── Save adapter ──────────────────────────────────────────────────
        save_adapter(trainer, output_dir, stage)

        # ── Merge (except Stage 10) ───────────────────────────────────────
        if stage not in NEVER_MERGE:
            # Free training VRAM before merge
            del model
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            merge_adapter(output_dir, stage)
        else:
            logger.info(f"Stage {stage} adapter retained as hot-swap LoRA. Not merging.")

        # ── Save completion metadata ──────────────────────────────────────
        trainer_type = "orpo" if is_orpo else ("dpo" if is_dpo else "sft")
        meta = {
            "stage": stage,
            "completed_at": datetime.now().isoformat(),
            "base_model": base_path,
            "examples_trained": len(dataset),
            "trainer_type": trainer_type,
            "is_dpo": is_dpo,
            "is_orpo": is_orpo,
            "merged": stage not in NEVER_MERGE,
        }
        (output_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))

        # ── Disk cleanup: remove previous stage's merged model + old checkpoints ──
        self._cleanup_previous_stages(stage)

        logger.info(f"Stage {stage} COMPLETE ✅")

    def _cleanup_previous_stages(self, current_stage: str):
        """
        Delete merged models from stages OLDER than the current one, but
        KEEP the current stage's merged model (it's the base for the next stage).
        Also delete checkpoints from all completed stages to save disk space.
        Each merged model is ~15GB; we only need the latest one.
        """
        idx = FULL_STAGE_ORDER.index(current_stage)
        for prev_idx in range(idx):
            prev_stage = FULL_STAGE_ORDER[prev_idx]
            prev_dir = self._stage_output_dir(prev_stage)

            # Delete merged model from stages BEFORE the current one
            # (current stage's merged IS the new base — must keep it!)
            merged_dir = prev_dir / "merged"
            if merged_dir.exists():
                size_mb = sum(f.stat().st_size for f in merged_dir.rglob("*") if f.is_file()) / 1_048_576
                shutil.rmtree(merged_dir)
                logger.info(f"♻️  Disk cleanup: removed {prev_stage}/merged/ ({size_mb:.0f} MB freed)")

            # Delete adapters from stages older than the immediate previous
            # Keep the most recent adapter as a fallback
            if prev_idx < idx - 1:
                adapter_dir = prev_dir / "adapter"
                if adapter_dir.exists():
                    size_mb = sum(f.stat().st_size for f in adapter_dir.rglob("*") if f.is_file()) / 1_048_576
                    shutil.rmtree(adapter_dir)
                    logger.info(f"♻️  Disk cleanup: removed {prev_stage}/adapter/ ({size_mb:.0f} MB freed)")

            # Delete checkpoints from all completed stages
            ckpt_dir = prev_dir / "checkpoints"
            if ckpt_dir.exists() and any(ckpt_dir.iterdir()):
                size_mb = sum(f.stat().st_size for f in ckpt_dir.rglob("*") if f.is_file()) / 1_048_576
                shutil.rmtree(ckpt_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"♻️  Disk cleanup: removed {prev_stage}/checkpoints/ ({size_mb:.0f} MB freed)")

    def run_all(self):
        """Run all 10 base stages in order."""
        import gc
        logger.info("Starting full 10-stage curriculum training...")
        logger.info(f"Output: {self.output_root.resolve()}")

        for i, stage in enumerate(STAGE_ORDER):
            logger.info(f"\n[{i+1}/{len(STAGE_ORDER)}] {stage}")
            try:
                self.run_stage(stage)
            except Exception as e:
                logger.error(f"Stage {stage} failed: {e}")
                logger.error("Fix the error and resume with: --all --resume")
                raise
            finally:
                # Full VRAM + RAM purge between stages
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        logger.info("\n🎉 All 10 base stages complete! Model ready at: fine_tuned/")
        print_status()

    def run_extended(self):
        """Run extended stages 11-15 (after base 10 are complete)."""
        import gc

        # Verify base stages are complete
        for stage in STAGE_ORDER:
            if not self._is_complete(stage):
                logger.error(
                    f"Base stage {stage} is not complete. "
                    f"Run --all first to complete stages 1-10 before running --extended."
                )
                sys.exit(1)

        logger.info("Starting extended 5-stage training (stages 11-15)...")
        logger.info(f"Output: {self.output_root.resolve()}")

        for i, stage in enumerate(EXTENDED_STAGE_ORDER):
            logger.info(f"\n[{i+1}/{len(EXTENDED_STAGE_ORDER)}] {stage}")
            try:
                self.run_stage(stage)
            except Exception as e:
                logger.error(f"Stage {stage} failed: {e}")
                logger.error("Fix the error and resume with: --extended --resume")
                raise
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        logger.info("\n🎉 All 15 stages complete! Full model ready at: fine_tuned/")
        print_status()

    def run_full(self):
        """Run all 15 stages (base + extended)."""
        import gc
        logger.info("Starting full 15-stage curriculum training...")
        logger.info(f"Output: {self.output_root.resolve()}")

        for i, stage in enumerate(FULL_STAGE_ORDER):
            logger.info(f"\n[{i+1}/{len(FULL_STAGE_ORDER)}] {stage}")
            try:
                self.run_stage(stage)
            except Exception as e:
                logger.error(f"Stage {stage} failed: {e}")
                logger.error("Fix the error and resume with: --full --resume")
                raise
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        logger.info("\n🎉 All 15 stages complete! Full model ready at: fine_tuned/")
        print_status()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cortex Lab — 15-Stage Curriculum Fine-Tuning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/fine_tune_cortex.py --status
  python scripts/fine_tune_cortex.py --stage stage1_faithfulness
  python scripts/fine_tune_cortex.py --all                 # Stages 1-10
  python scripts/fine_tune_cortex.py --all --resume
  python scripts/fine_tune_cortex.py --extended             # Stages 11-15
  python scripts/fine_tune_cortex.py --extended --resume
  python scripts/fine_tune_cortex.py --full                 # All 15 stages
  python scripts/fine_tune_cortex.py --full --resume
  python scripts/fine_tune_cortex.py --merge-only stage1_faithfulness
  python scripts/fine_tune_cortex.py --validate-data

Base Stages (1-10):
  1. stage1_faithfulness   — RAG-grounded citation behavior
  2. stage2_agentic        — Tool-use and JSON routing
  3. stage3_causal         — Temporal causal reasoning
  4. stage4_selfrag        — Self-critique (ISREL/ISSUP/ISUSE)
  5. stage5_belief         — Belief evolution tracking
  6. stage6_summarization  — Memory compression
  7. stage7_dialogue       — Multi-turn coherence
  8. stage8_longcontext    — Long-context multi-hop reasoning
  9. stage9_dpo            — Preference alignment (DPO)
  10. stage10_user_style   — Personal style adapter (hot-swap)

Extended Stages (11-15):
  11. stage11_orpo            — ORPO preference optimization
  12. stage12_raft            — Retrieval-augmented fine-tuning
  13. stage13_function_calling — Structured function calling
  14. stage14_rft             — Rejection sampling (best-of-N)
  15. stage15_spin            — Self-play improvement (SPIN)
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--stage", type=str, metavar="STAGE",
        help="Train a single stage (e.g., stage1_faithfulness, stage11_orpo)"
    )
    group.add_argument(
        "--all", action="store_true",
        help="Train base stages 1-10 sequentially"
    )
    group.add_argument(
        "--extended", action="store_true",
        help="Train extended stages 11-15 (requires base 1-10 complete)"
    )
    group.add_argument(
        "--full", action="store_true",
        help="Train all 15 stages sequentially"
    )
    group.add_argument(
        "--status", action="store_true",
        help="Show training status for all stages"
    )
    group.add_argument(
        "--merge-only", type=str, metavar="STAGE",
        help="Only merge the adapter for a completed stage"
    )
    group.add_argument(
        "--validate-data", action="store_true",
        help="Validate all dataset files without training"
    )

    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from the latest checkpoint"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force retrain even if stage is already complete"
    )

    return parser.parse_args()


def validate_datasets():
    """Validate all dataset files exist and are well-formed."""
    print("\n" + "=" * 60)
    print("  Validating datasets...")
    print("=" * 60)
    all_ok = True
    total_examples = 0

    for stage, filename in STAGE_DATA_FILES.items():
        path = Path(TRAINING_DATA_DIR) / filename
        if not path.exists():
            print(f"  ❌ MISSING: {filename}")
            all_ok = False
            continue

        try:
            with open(path, "r") as f:
                data = json.load(f)

            is_dpo = stage in DPO_STAGES
            is_orpo = stage in ORPO_STAGES
            is_preference = is_dpo or is_orpo
            valid = 0
            for ex in data:
                if is_preference:
                    if ex.get("prompt") and ex.get("chosen") and ex.get("rejected"):
                        valid += 1
                else:
                    if ex.get("instruction") and ex.get("output"):
                        valid += 1

            size_mb = path.stat().st_size / 1_048_576
            trainer = "ORPO" if is_orpo else ("DPO" if is_dpo else "SFT")
            print(f"  ✅ {filename:<45} {valid:>5,} valid / {len(data):>5,} total  ({size_mb:.1f} MB) [{trainer}]")
            total_examples += valid

        except Exception as e:
            print(f"  ❌ ERROR: {filename}: {e}")
            all_ok = False

    print("=" * 60)
    print(f"  Total valid examples: {total_examples:,}")
    print(f"  Status: {'✅ ALL OK' if all_ok else '❌ ISSUES FOUND'}")
    print("=" * 60 + "\n")
    return all_ok


def main():
    args = parse_args()

    # ── Status / validation — no heavy imports needed ───────────────────────
    if args.status:
        # Lazy-import just torch for VRAM info
        try:
            import torch
            globals()["torch"] = torch
            check_vram()
        except ImportError:
            pass
        print_status()
        return

    if args.validate_data:
        ok = validate_datasets()
        sys.exit(0 if ok else 1)

    if args.merge_only:
        _import_training_deps()
        check_vram()
        merge_adapter(Path(OUTPUT_DIR) / args.merge_only, args.merge_only)
        return

    # ── Training — load all heavy deps ──────────────────────────────────────
    _import_training_deps()
    check_vram()

    trainer = CortexLabTrainer(resume=args.resume)

    if args.all:
        trainer.run_all()
    elif args.extended:
        trainer.run_extended()
    elif args.full:
        trainer.run_full()
    elif args.stage:
        trainer.run_stage(args.stage)


if __name__ == "__main__":
    main()
