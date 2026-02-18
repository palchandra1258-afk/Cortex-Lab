# Cortex Lab - DeepSeek-R1-1.5B Setup

This repository contains setup scripts and tools for Cortex Lab, powered by DeepSeek-R1-1.5B, a lightweight 1.5B parameter reasoning model.

## Model Information

- **Model**: DeepSeek-R1-1.5B
- **Parameters**: 1.5 Billion
- **Base Model**: DeepSeek-R1
- **Context Length**: 32K tokens
- **License**: MIT
- **Use Cases**: Reasoning tasks, mathematics, coding, instruction following

## Features

- ✅ Easy model download and setup
- ✅ Lightweight model (runs on most GPUs)
- ✅ Interactive chat interface
- ✅ Batch inference
- ✅ Full customization

## Requirements

### Minimum Hardware
- **RAM**: 8GB+
- **GPU**: 4GB+ VRAM (or CPU)
- **Disk**: 10GB free space

### Recommended Hardware
- **GPU**: NVIDIA RTX 3090/4090 or A100
- **RAM**: 64GB+
- **Disk**: SSD with 50GB+ free space

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Model

**Option A: Full precision (requires ~30GB VRAM)**
```bash
python setup_model.py
```

**Option B: 4-bit quantization (requires ~8GB VRAM)**
```bash
python setup_model.py --4bit
```

**Option C: 8-bit quantization (requires ~14GB VRAM)**
```bash
python setup_model.py --8bit
```

**Skip inference test:**
```bash
python setup_model.py --skip-test
```

## Usage

### Interactive Chat

```bash
# Standard inference
python inference.py

# With 4-bit quantization
python inference.py --4bit

# With 8-bit quantization
python inference.py --8bit

# From local model path
python inference.py --model-name ./models/DeepSeek-R1-Distill-Qwen-14B
```

### Batch Inference

```bash
# Single prompt
python inference.py --prompt "What is the derivative of x^2?"

# With custom parameters
python inference.py \
    --prompt "Solve this problem: 2x + 5 = 15" \
    --temperature 0.6 \
    --max-tokens 1024 \
    --4bit
```

### Fine-tuning / Training

**Basic LoRA fine-tuning:**
```bash
python train_model.py \
    --use-lora \
    --4bit \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-4
```

**With custom dataset:**
```bash
python train_model.py \
    --use-lora \
    --4bit \
    --dataset "your-dataset-name" \
    --epochs 5 \
    --batch-size 2
```

**Full fine-tuning (requires large GPU):**
```bash
python train_model.py \
    --epochs 3 \
    --batch-size 1 \
    --learning-rate 1e-5
```

**With Weights & Biases logging:**
```bash
python train_model.py \
    --use-lora \
    --4bit \
    --use-wandb
```

## Usage Recommendations

Based on the official DeepSeek-R1 documentation:

1. **Temperature**: Use 0.5-0.7 range (0.6 recommended) to prevent repetition
2. **System Prompts**: Avoid system prompts; put all instructions in user prompt
3. **Math Problems**: Include directive like "Please reason step by step, and put your final answer within \\boxed{}"
4. **Thinking Pattern**: The model uses `<think>` tags for reasoning - enforce this for best results
5. **Multiple Runs**: For evaluation, run multiple times and average results

## Training Arguments

### LoRA Configuration
- `r=16`: LoRA rank (higher = more parameters)
- `lora_alpha=32`: Scaling factor
- `lora_dropout=0.05`: Dropout for LoRA layers
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### Training Parameters
- `learning_rate=2e-4`: Learning rate for LoRA
- `learning_rate=1e-5`: Learning rate for full fine-tuning
- `batch_size=4`: Adjust based on GPU memory
- `gradient_accumulation_steps=4`: Effective batch size multiplier
- `warmup_steps=100`: Linear warmup steps
- `max_length=2048`: Maximum sequence length

## Advanced Usage

### Using vLLM for Fast Inference

```bash
# Install vLLM
pip install vllm

# Start server
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --tensor-parallel-size 1 \
    --max-model-len 8192
```

### Using SGLang

```bash
# Install SGLang
pip install sglang

# Start server
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --trust-remote-code \
    --tp 1
```

## File Structure

```
DeepLearning/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── setup_model.py            # Model download and setup
├── train_model.py            # Training/fine-tuning script
├── inference.py              # Inference script
├── models/                   # Downloaded models
│   └── DeepSeek-R1-Distill-Qwen-14B/
├── output/                   # Training outputs
│   ├── final_model/         # Final trained model
│   └── logs/                # Training logs
├── checkpoints/             # Training checkpoints
└── cache/                   # HuggingFace cache
```

## Performance Benchmarks

From official documentation:

| Benchmark | Score |
|-----------|-------|
| AIME 2024 | 69.7% |
| MATH-500 | 93.9% |
| Codeforces | 1481 rating |
| MMLU | 90.8% |
| GPQA-Diamond | 59.1% |

## Troubleshooting

### Out of Memory Errors
- Use 4-bit quantization: `--4bit`
- Reduce batch size: `--batch-size 1`
- Enable gradient accumulation
- Use smaller max sequence length

### Slow Download
- Check internet connection
- Use HuggingFace mirror: `HF_ENDPOINT=https://hf-mirror.com`
- Download with `huggingface-cli download`

### Model Not Found
- Ensure git-lfs is installed: `sudo apt-get install git-lfs`
- Clear cache: `rm -rf ~/.cache/huggingface`
- Try manual download from HuggingFace website

## Resources

- **Model Card**: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
- **Paper**: https://arxiv.org/abs/2501.12948
- **Official Website**: https://www.deepseek.com/
- **Chat Interface**: https://chat.deepseek.com/
- **API**: https://platform.deepseek.com/

## License

This code is provided under MIT License. The model is also under MIT License, allowing commercial use.

Note: This model is derived from Qwen2.5-14B (Apache 2.0 License).

## Citation

```bibtex
@misc{deepseekai2025deepseekr1,
  title={DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning}, 
  author={DeepSeek-AI},
  year={2025},
  eprint={2501.12948},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
}
```

## Support

For issues and questions:
- GitHub Issues: [Create an issue]
- Email: service@deepseek.com
- Discord: https://discord.gg/Tc7c45Zzu5
