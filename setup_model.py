"""
Setup script for DeepSeek-R1-Distill-Qwen-14B model
This script downloads and prepares the model for training/inference
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
LOCAL_MODEL_PATH = "./models/DeepSeek-R1-Distill-Qwen-14B"
CACHE_DIR = "./cache"

def check_gpu_availability():
    """Check if GPU is available and print information"""
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("⚠ No GPU detected. Model will run on CPU (slower)")
        return False

def download_model(use_4bit=False, use_8bit=False):
    """
    Download and cache the model
    
    Args:
        use_4bit: Use 4-bit quantization (requires bitsandbytes)
        use_8bit: Use 8-bit quantization (requires bitsandbytes)
    """
    print(f"\n{'='*60}")
    print(f"Downloading DeepSeek-R1-Distill-Qwen-14B")
    print(f"{'='*60}\n")
    
    # Create directories
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
    
    # Check GPU
    has_gpu = check_gpu_availability()
    
    # Configure model loading parameters
    load_kwargs = {
        "cache_dir": CACHE_DIR,
        "trust_remote_code": True,
    }
    
    # Add quantization if requested
    if use_4bit:
        from transformers import BitsAndBytesConfig
        print("\n✓ Using 4-bit quantization")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif use_8bit:
        print("\n✓ Using 8-bit quantization")
        load_kwargs["load_in_8bit"] = True
    elif has_gpu:
        load_kwargs["torch_dtype"] = torch.bfloat16
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.float32
    
    try:
        # Download tokenizer
        print("\n[1/2] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            trust_remote_code=True
        )
        print("✓ Tokenizer downloaded successfully")
        
        # Download model
        print("\n[2/2] Downloading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            **load_kwargs
        )
        print("✓ Model downloaded successfully")
        
        # Save locally
        print(f"\n[3/3] Saving model to {LOCAL_MODEL_PATH}...")
        tokenizer.save_pretrained(LOCAL_MODEL_PATH)
        if not (use_4bit or use_8bit):
            model.save_pretrained(LOCAL_MODEL_PATH)
        
        print(f"\n{'='*60}")
        print("✓ Setup Complete!")
        print(f"{'='*60}")
        print(f"\nModel saved to: {LOCAL_MODEL_PATH}")
        print(f"Cache directory: {CACHE_DIR}")
        
        # Print model info
        print(f"\nModel Information:")
        print(f"  - Model size: ~14B parameters")
        print(f"  - Base model: Qwen2.5-14B")
        print(f"  - Max context length: 32K tokens")
        print(f"  - License: MIT")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nTroubleshooting tips:")
        print("  1. Check your internet connection")
        print("  2. Ensure you have enough disk space (~30GB)")
        print("  3. Try installing git-lfs: sudo apt-get install git-lfs")
        print("  4. Verify huggingface-hub is installed: pip install huggingface-hub")
        raise

def test_inference(model, tokenizer):
    """Test basic inference"""
    print(f"\n{'='*60}")
    print("Testing Inference")
    print(f"{'='*60}\n")
    
    # Test prompt
    prompt = "What is the sum of 123 and 456? Please reason step by step, and put your final answer within \\boxed{}."
    
    print(f"Prompt: {prompt}\n")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate
    print("Generating response...\n")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response:\n{response}\n")
    print("✓ Inference test successful!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup DeepSeek-R1-Distill-Qwen-14B")
    parser.add_argument("--4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--skip-test", action="store_true", help="Skip inference test")
    
    args = parser.parse_args()
    
    # Download model
    model, tokenizer = download_model(use_4bit=args.__dict__['4bit'], use_8bit=args.__dict__['8bit'])
    
    # Test inference
    if not args.skip_test and not args.__dict__['4bit'] and not args.__dict__['8bit']:
        try:
            test_inference(model, tokenizer)
        except Exception as e:
            print(f"\n⚠ Inference test failed: {e}")
            print("Model is downloaded but you may need to troubleshoot inference.")
