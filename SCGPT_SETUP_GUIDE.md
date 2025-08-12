# scGPT Virtual Environment Setup Guide

This guide will help you set up scGPT with a new virtual environment and all necessary dependencies.

## Quick Setup

### Option 1: Automated Setup (Recommended)

1. **Run the setup script:**
   ```bash
   ./setup_scgpt_venv.sh
   ```

2. **Activate the environment:**
   ```bash
   source activate_scgpt_venv.sh
   ```

### Option 2: Manual Setup

1. **Create virtual environment:**
   ```bash
   python3 -m venv scgpt_venv
   source scgpt_venv/bin/activate
   ```

2. **Set up cache directories:**
   ```bash
   mkdir -p ./cache/{pip,huggingface,transformers,matplotlib,torch,tmp,.xdg,egg}
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements_scgpt.txt
   ```

4. **Clone and install Geneformer:**
   ```bash
   git lfs install
   git clone https://huggingface.co/ctheodoris/Geneformer
   pip install ./Geneformer
   ```

## Environment Variables

The setup automatically configures these environment variables for local caching:

- `PIP_CACHE_DIR`: `./cache/pip`
- `HF_HOME`: `./cache/huggingface`
- `TRANSFORMERS_CACHE`: `./cache/transformers`
- `MPLCONFIGDIR`: `./cache/matplotlib`
- `TORCH_HOME`: `./cache/torch`
- `TMPDIR`: `./cache/tmp`
- `XDG_CACHE_HOME`: `./cache/.xdg`
- `PYTHON_EGG_CACHE`: `./cache/egg`

## Python Setup

To set up cache directories in Python, run:
```python
python setup_cache.py
```

## Key Dependencies

- **PyTorch**: 1.13.0 (CPU version for compatibility)
- **scGPT**: 0.1.6
- **Transformers**: 4.35.2
- **Anndata**: 0.9.2
- **Scanpy**: Latest
- **Matplotlib**: 3.7.1

## Troubleshooting

### Common Issues

1. **Git LFS not found:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install git-lfs
   
   # CentOS/RHEL
   sudo yum install git-lfs
   ```

2. **Disk quota exceeded:**
   - The setup uses local cache directories to avoid quota issues
   - All downloads are cached locally in `./cache/`

3. **PyTorch installation issues:**
   - The script installs CPU version for maximum compatibility
   - If you need GPU support, modify the PyTorch installation line

4. **Transformers version conflicts:**
   - The script installs transformers==4.35.2 which is compatible with scGPT
   - Do not upgrade transformers without testing

### Verification

To verify the setup, run:
```python
import torch
import transformers
import scgpt
import scanpy

print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print("All imports successful!")
```

## Usage

1. **Activate the environment:**
   ```bash
   source activate_scgpt_venv.sh
   ```

2. **Set up cache in Python:**
   ```python
   import os
   os.environ["TRANSFORMERS_CACHE"] = "./cache/transformers"
   os.environ["HF_HOME"] = "./cache/huggingface"
   ```

3. **Run your scGPT notebooks:**
   ```bash
   jupyter notebook
   ```

## File Structure

After setup, you'll have:
```
zero-shot-scfoundation/
├── scgpt_venv/           # Virtual environment
├── cache/                 # Local cache directories
├── Geneformer/           # Geneformer repository
├── activate_scgpt_venv.sh # Activation script
├── setup_cache.py        # Python cache setup
└── requirements_scgpt.txt # Dependencies list
```

## Notes

- The setup uses CPU PyTorch for maximum compatibility
- All downloads are cached locally to avoid quota issues
- The environment is isolated and won't interfere with other projects
- Cache directories are created automatically 