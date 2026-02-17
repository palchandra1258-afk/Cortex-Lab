#!/bin/bash

# Installation script with virtual environment setup

set -e

echo "=========================================="
echo "DeepSeek-R1-Distill-Qwen-14B Installation"
echo "=========================================="
echo ""

cd /home/vsu/Downloads/DeepLearning

# Create virtual environment
echo "[1/4] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "[2/4] Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "[3/4] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo ""
echo "[4/4] Installing dependencies..."
pip install torch transformers accelerate peft bitsandbytes datasets trl optimum safetensors sentencepiece protobuf tokenizers numpy pandas scipy tqdm huggingface-hub

echo ""
echo "=========================================="
echo "✓ Installation Complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Then you can:"
echo "  1. Download the model: python setup_model.py --4bit"
echo "  2. Run inference: python inference.py --4bit"
echo "  3. Train/fine-tune: python train_model.py --use-lora --4bit"
echo ""
