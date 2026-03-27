#!/bin/bash

set -e  # Stop on error

ENV_NAME="bioner"
REQ_FILE="/home/mkeber/syn-bioner/bioNER/start-bioner-docker/requirements_base.txt"

echo "======================================"
echo "Creating conda environment: $ENV_NAME"
echo "======================================"

conda create -n $ENV_NAME python=3.10 -y

echo "======================================"
echo "Activating environment"
echo "======================================"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "======================================"
echo "Installing PyTorch with CUDA 12.1"
echo "======================================"

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

echo "======================================"
echo "Upgrading pip"
echo "======================================"

pip install --upgrade pip

echo "======================================"
echo "Installing project requirements"
echo "======================================"

pip install -r $REQ_FILE

echo "======================================"
echo "Installing spaCy with CUDA support"
echo "======================================"

pip install spacy[cuda12x]

echo "======================================"
echo "Downloading spaCy English model"
echo "======================================"

if python -m spacy download en_core_web_trf; then
    echo "Installed en_core_web_trf"
else
    echo "Transformer model not available, installing en_core_web_lg"
    python -m spacy download en_core_web_lg
fi

echo "======================================"
echo "Downloading spaCy German model"
echo "======================================"

if python -m spacy download de_core_news_trf; then
    echo "Installed de_core_news_trf"
else
    echo "Transformer model not available, installing de_core_news_lg"
    python -m spacy download de_core_news_lg
fi

echo "======================================"
echo "Verifying GPU availability"
echo "======================================"

python - <<EOF
import torch
import spacy

print("CUDA available (torch):", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("spaCy prefers GPU:", spacy.prefer_gpu())
EOF

echo "======================================"
echo "Installation complete!"
echo "Activate with: conda activate $ENV_NAME"
echo "======================================"