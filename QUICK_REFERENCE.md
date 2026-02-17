# DeepSeek-R1-Distill-Qwen-14B - Quick Reference

## 🚀 Quick Start (3 Steps)

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Download model (with 4-bit quantization)
python setup_model.py --4bit

# 3. Start chatting!
python inference.py --4bit
```

## 📋 Common Commands

### Model Download
```bash
python setup_model.py --4bit        # Recommended: ~8GB VRAM
python setup_model.py --8bit        # Alternative: ~14GB VRAM  
python setup_model.py               # Full precision: ~30GB VRAM
```

### Interactive Chat
```bash
python inference.py --4bit          # Start chat with 4-bit model
python inference.py --8bit          # Start chat with 8-bit model
```

### Single Inference
```bash
python inference.py --4bit --prompt "Your question here"
python inference.py --4bit --prompt "Solve: 2x + 5 = 13" --max-tokens 1024
```

### Training/Fine-tuning
```bash
# LoRA fine-tuning (most efficient)
python train_model.py --use-lora --4bit --epochs 3

# With custom dataset
python train_model.py --use-lora --4bit --dataset "username/dataset" --epochs 5

# With W&B logging
python train_model.py --use-lora --4bit --use-wandb
```

## 🎯 Model Specifications

- **Parameters**: 14 Billion
- **Base Model**: Qwen2.5-14B
- **Context Length**: 32,768 tokens
- **License**: MIT (Commercial use allowed)
- **Specialties**: Reasoning, Math, Coding

## 💡 Best Practices

1. **Temperature**: Use 0.6 (range: 0.5-0.7)
2. **Math Problems**: Ask to put answer in `\boxed{}`
3. **Prompting**: Avoid system prompts, use user prompts only
4. **Thinking**: Model uses `<think>` tags - this is expected behavior
5. **Benchmarking**: Run multiple times and average results

## 🔧 Troubleshooting

### Out of Memory
```bash
# Use 4-bit quantization
python setup_model.py --4bit

# Or reduce batch size when training
python train_model.py --use-lora --4bit --batch-size 1
```

### Slow Download
```bash
# Install git-lfs
sudo apt-get install git-lfs

# Clear cache if needed
rm -rf ~/.cache/huggingface
```

### Model Not Found
```bash
# Verify virtual environment is activated
source venv/bin/activate

# Check installation
pip list | grep transformers
```

## 📊 Performance Benchmarks

| Benchmark | Score |
|-----------|-------|
| AIME 2024 | 69.7% |
| MATH-500  | 93.9% |
| MMLU      | 90.8% |
| Codeforces| 1481  |

## 📁 File Structure

```
DeepLearning/
├── setup_model.py      # Download & setup model
├── inference.py        # Run inference/chat
├── train_model.py      # Fine-tune model
├── requirements.txt    # Python dependencies
├── README.md          # Full documentation
├── install.sh         # Installation script
├── USAGE_GUIDE.sh     # Detailed usage guide
└── venv/             # Virtual environment
```

## 🔗 Resources

- **Model Page**: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
- **Paper**: https://arxiv.org/abs/2501.12948
- **Website**: https://www.deepseek.com/
- **Discord**: https://discord.gg/Tc7c45Zzu5

## 💬 Example Prompts

```python
# Math problem
"Solve the equation 3x + 7 = 22. Please reason step by step and put your final answer within \boxed{}."

# Coding question
"Write a Python function to find the nth Fibonacci number using dynamic programming."

# Reasoning task
"A farmer has 17 sheep. All but 9 die. How many are left? Think through this carefully."
```

## 🎓 Training Tips

**LoRA Parameters:**
- `r=16`: LoRA rank (increase for more capacity)
- `lora_alpha=32`: Scaling factor
- `learning_rate=2e-4`: Good starting point for LoRA

**Training Strategy:**
1. Start with small learning rate (2e-4)
2. Use 4-bit quantization to save memory
3. Monitor loss on validation set
4. Save checkpoints every 500 steps

---

**Need help?** Check `README.md` or run `./USAGE_GUIDE.sh`
