#!/bin/bash

# scGPT Virtual Environment Setup Script
# This script sets up a new virtual environment for scGPT with all necessary dependencies

set -e  # Exit on any error

echo "Setting up scGPT virtual environment..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv scgpt_venv

# Activate virtual environment
echo "Activating virtual environment..."
source scgpt_venv/bin/activate

# Set up cache directories locally
echo "Setting up cache directories..."
export PIP_CACHE_DIR="./cache/pip"
export HF_HOME="./cache/huggingface"
export TRANSFORMERS_CACHE="./cache/transformers"
export MPLCONFIGDIR="./cache/matplotlib"
export TORCH_HOME="./cache/torch"
export TMPDIR="./cache/tmp"
export XDG_CACHE_HOME="./cache/.xdg"
export PYTHON_EGG_CACHE="./cache/egg"

# Create cache directories
mkdir -p ./cache/pip
mkdir -p ./cache/huggingface
mkdir -p ./cache/transformers
mkdir -p ./cache/matplotlib
mkdir -p ./cache/torch
mkdir -p ./cache/tmp
mkdir -p ./cache/.xdg
mkdir -p ./cache/egg

echo "Cache directories created:"
echo "PIP_CACHE_DIR: $PIP_CACHE_DIR"
echo "HF_HOME: $HF_HOME"
echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "MPLCONFIGDIR: $MPLCONFIGDIR"
echo "TORCH_HOME: $TORCH_HOME"
echo "TMPDIR: $TMPDIR"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (CPU version for compatibility)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install core dependencies
echo "Installing core dependencies..."
pip install matplotlib==3.7.1
pip install tqdm
pip install seaborn
pip install anndata==0.9.2
pip install colorlog==6.7.0
pip install numpy
pip install pandas
pip install scanpy
pip install scipy
pip install scib
pip install scikit-learn

# Install scGPT
echo "Installing scGPT..."
pip install scgpt==0.2.4

# Install transformers with specific version
echo "Installing transformers..."
pip install transformers==4.35.2

# Install torchtext
echo "Installing torchtext..."
pip install torchtext==0.17.0

# Install additional dependencies
echo "Installing additional dependencies..."
pip install PyComplexHeatmap
pip install scvi-tools
pip install geneformer==0.0.1

# Install Git LFS if not available
if ! command -v git-lfs &> /dev/null; then
    echo "Git LFS not found. Please install it manually:"
    echo "  Ubuntu/Debian: sudo apt-get install git-lfs"
    echo "  CentOS/RHEL: sudo yum install git-lfs"
    echo "  Or download from: https://git-lfs.github.com/"
fi

# Clone Geneformer repository if not exists
if [ ! -d "Geneformer" ]; then
    echo "Cloning Geneformer repository..."
    git lfs install
    git clone https://huggingface.co/ctheodoris/Geneformer
    pip install ./Geneformer
else
    echo "Geneformer directory already exists. Installing..."
    pip install ./Geneformer
fi

# Create activation script for future use
cat > activate_scgpt_venv.sh << 'EOF'
#!/bin/bash

# Activate scGPT virtual environment
source scgpt_venv/bin/activate

# Set up cache directories
export PIP_CACHE_DIR="./cache/pip"
export HF_HOME="./cache/huggingface"
export TRANSFORMERS_CACHE="./cache/transformers"
export MPLCONFIGDIR="./cache/matplotlib"
export TORCH_HOME="./cache/torch"
export TMPDIR="./cache/tmp"
export XDG_CACHE_HOME="./cache/.xdg"
export PYTHON_EGG_CACHE="./cache/egg"

echo "scGPT virtual environment activated!"
echo "Cache directories set up:"
echo "PIP_CACHE_DIR: $PIP_CACHE_DIR"
echo "HF_HOME: $HF_HOME"
echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "MPLCONFIGDIR: $MPLCONFIGDIR"
echo "TORCH_HOME: $TORCH_HOME"
echo "TMPDIR: $TMPDIR"

# Keep the shell open
exec bash
EOF

chmod +x activate_scgpt_venv.sh

# Create Python setup script
cat > setup_cache.py << 'EOF'
import os

# Set up cache directories for local storage
cache_dirs = [
    "./cache/pip",
    "./cache/huggingface", 
    "./cache/transformers",
    "./cache/matplotlib",
    "./cache/torch",
    "./cache/tmp",
    "./cache/.xdg",
    "./cache/egg"
]

# Set environment variables
os.environ["PIP_CACHE_DIR"] = "./cache/pip"
os.environ["HF_HOME"] = "./cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "./cache/transformers"
os.environ["MPLCONFIGDIR"] = "./cache/matplotlib"
os.environ["TORCH_HOME"] = "./cache/torch"
os.environ["TMPDIR"] = "./cache/tmp"
os.environ["XDG_CACHE_HOME"] = "./cache/.xdg"
os.environ["PYTHON_EGG_CACHE"] = "./cache/egg"

# Create cache directories if they don't exist
for cache_dir in cache_dirs:
    os.makedirs(cache_dir, exist_ok=True)

print("Cache directories set up:")
print(f"PIP_CACHE_DIR: {os.environ.get('PIP_CACHE_DIR')}")
print(f"HF_HOME: {os.environ.get('HF_HOME')}")
print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE')}")
print(f"MPLCONFIGDIR: {os.environ.get('MPLCONFIGDIR')}")
print(f"TORCH_HOME: {os.environ.get('TORCH_HOME')}")
print(f"TMPDIR: {os.environ.get('TMPDIR')}")
EOF

echo ""
echo "=========================================="
echo "scGPT Virtual Environment Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future:"
echo "  source activate_scgpt_venv.sh"
echo ""
echo "Or manually:"
echo "  source scgpt_venv/bin/activate"
echo ""
echo "To set up cache directories in Python:"
echo "  python setup_cache.py"
echo ""
echo "The environment includes:"
echo "- PyTorch 1.13.0 (CPU version)"
echo "- scGPT 0.1.6"
echo "- Transformers 4.35.2"
echo "- All required dependencies"
echo "- Local cache directories"
echo ""
echo "You can now run your scGPT notebooks!" 