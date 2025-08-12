#!/bin/bash

echo "Setting up new virtual environment with local cache directories..."

# Activate the virtual environment
source new_venv/bin/activate

# Set up cache directories locally
export PIP_CACHE_DIR="./cache/pip"
export HF_HOME="./cache/huggingface"
export TRANSFORMERS_CACHE="./cache/transformers"
export MPLCONFIGDIR="./cache/matplotlib"

# Create cache directories if they don't exist
mkdir -p ./cache/pip
mkdir -p ./cache/huggingface
mkdir -p ./cache/transformers
mkdir -p ./cache/matplotlib

echo "Virtual environment activated with local cache directories:"
echo "PIP_CACHE_DIR: $PIP_CACHE_DIR"
echo "HF_HOME: $HF_HOME"
echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "MPLCONFIGDIR: $MPLCONFIGDIR"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "Virtual environment is ready!"
echo "To activate this environment in the future, run:"
echo "source new_venv/bin/activate"
echo ""
echo "To set up cache directories in Python, run:"
echo "python setup_cache.py" 