#!/bin/bash

# Quick Start Script for DeepSeek-R1-Distill-Qwen-14B
# This script helps you get started quickly

set -e

echo "=========================================="
echo "DeepSeek-R1-Distill-Qwen-14B Quick Start"
echo "=========================================="
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check CUDA availability
echo ""
echo "[2/5] Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ No NVIDIA GPU detected. Model will run on CPU (slower)"
fi

# Install dependencies
echo ""
echo "[3/5] Installing dependencies..."
read -p "Install dependencies from requirements.txt? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "⚠ Skipped dependency installation"
fi

# Download model
echo ""
echo "[4/5] Model download options:"
echo "  1) Full precision (~30GB VRAM required)"
echo "  2) 4-bit quantization (~8GB VRAM required) [RECOMMENDED]"
echo "  3) 8-bit quantization (~14GB VRAM required)"
echo "  4) Skip download (already downloaded)"
echo ""
read -p "Select option (1-4): " -n 1 -r
echo

case $REPLY in
    1)
        echo "Downloading model in full precision..."
        python setup_model.py
        ;;
    2)
        echo "Downloading model with 4-bit quantization..."
        python setup_model.py --4bit
        ;;
    3)
        echo "Downloading model with 8-bit quantization..."
        python setup_model.py --8bit
        ;;
    4)
        echo "⚠ Skipped model download"
        ;;
    *)
        echo "Invalid option. Skipping download."
        ;;
esac

# Next steps
echo ""
echo "[5/5] Setup complete!"
echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Interactive Chat:"
echo "   python inference.py --4bit"
echo ""
echo "2. Single Prompt:"
echo "   python inference.py --4bit --prompt 'What is 2+2?'"
echo ""
echo "3. Fine-tune Model:"
echo "   python train_model.py --use-lora --4bit --epochs 3"
echo ""
echo "4. Read Documentation:"
echo "   cat README.md"
echo ""
echo "=========================================="
echo "Happy coding! 🚀"
echo "=========================================="
