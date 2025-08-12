#!/bin/bash

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

# Keep the shell open
exec bash 