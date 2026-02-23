#!/usr/bin/env python3
"""
Recovery script: Rebuild stage 9 merged model by sequentially applying
adapters 5→6→7→8→9 on top of the base model.

Stages 1-4 were fully cleaned (no adapters/merged), so we start from
the base model. This is an approximation since each adapter was trained
on the previous stage's merged model, but since LoRA adapters are small
perturbations, the result should still be functional.

After this script, stage 11 (ORPO) can proceed from the reconstructed
stage 9 merged model.
"""

import os
import sys
import json
import gc
import torch
from pathlib import Path

# Set up path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = str(PROJECT_ROOT / "models" / "deepseek-r1-7b")
FINE_TUNED = PROJECT_ROOT / "fine_tuned"

# Stages that have adapters we can recover (in order)
ADAPTER_CHAIN = [
    "stage5_belief",
    "stage6_summarization",
    "stage7_dialogue",
    "stage8_longcontext",
    "stage9_dpo",
]

def main():
    print("=" * 70)
    print("CORTEX-LAB MODEL RECOVERY")
    print("Rebuilding stage 9 merged model from base + adapter chain")
    print("=" * 70)

    # Verify all adapters exist
    for stage in ADAPTER_CHAIN:
        adapter_dir = FINE_TUNED / stage / "adapter"
        if not adapter_dir.exists():
            print(f"❌ Missing adapter: {adapter_dir}")
            sys.exit(1)
        print(f"✅ Found adapter: {stage}")

    # Step 1: Load base model on CPU (bf16 for merge — no quantization)
    print(f"\n📦 Loading base model: {BASE_MODEL}")
    print("   (Full precision on CPU for clean merge)")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Step 2: Sequentially apply and merge each adapter
    for i, stage in enumerate(ADAPTER_CHAIN):
        adapter_dir = str(FINE_TUNED / stage / "adapter")
        print(f"\n🔧 [{i+1}/{len(ADAPTER_CHAIN)}] Applying adapter: {stage}")

        # Apply adapter
        model = PeftModel.from_pretrained(
            model,
            adapter_dir,
            is_trainable=False,
        )

        # Merge into base weights
        model = model.merge_and_unload()
        print(f"   ✅ Merged {stage}")

        # Memory cleanup
        gc.collect()

    # Step 3: Save the final merged model as stage 9's merged
    output_dir = FINE_TUNED / "stage9_dpo" / "merged"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving merged model → {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save recovery metadata
    meta = {
        "recovery": True,
        "base_model": BASE_MODEL,
        "adapter_chain": ADAPTER_CHAIN,
        "note": "Reconstructed by sequentially merging adapters 5-9 on base model. "
                "Stages 1-4 adapters were lost to cleanup.",
    }
    (output_dir / "recovery_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n🎉 Recovery complete! Stage 9 merged model saved.")
    print(f"   Path: {output_dir}")
    print(f"\n   Now run: python scripts/fine_tune_cortex.py --full --resume")
    print(f"   Stages 1-10 will be skipped, stage 11 (ORPO) will start.")

    # Cleanup
    del model
    gc.collect()


if __name__ == "__main__":
    main()
