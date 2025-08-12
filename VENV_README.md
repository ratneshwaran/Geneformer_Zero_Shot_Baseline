# New Virtual Environment Setup

This directory contains a new virtual environment (`new_venv`) with local cache directories configured to avoid using the global cache locations.

## Quick Start

### Option 1: Use the comprehensive setup script
```bash
./activate_new_venv.sh
```

This script will:
- Activate the virtual environment
- Set up local cache directories
- Upgrade pip
- Install all requirements from `requirements.txt`

### Option 2: Manual setup

1. **Activate the virtual environment:**
   ```bash
   source new_venv/bin/activate
   ```

2. **Set up cache directories:**
   ```bash
   ./setup_venv.sh
   ```
   
   Or in Python:
   ```python
   python setup_cache.py
   ```

## Cache Directory Structure

The following cache directories are created locally within the project:

```
./cache/
├── pip/              # pip package cache
├── huggingface/      # Hugging Face models and datasets
├── transformers/     # Transformers library cache
└── matplotlib/       # Matplotlib configuration
```

## Environment Variables

The following environment variables are set to point to local directories:

- `PIP_CACHE_DIR`: `./cache/pip`
- `HF_HOME`: `./cache/huggingface`
- `TRANSFORMERS_CACHE`: `./cache/transformers`
- `MPLCONFIGDIR`: `./cache/matplotlib`

## Usage

After activation, you can use the virtual environment normally. All cached files will be stored locally in the `./cache` directory instead of global system locations.

### Installing packages
```bash
pip install package_name
```

### Using in Python scripts
```python
import os

# Set up cache directories (if not already done)
os.environ["PIP_CACHE_DIR"] = "./cache/pip"
os.environ["HF_HOME"] = "./cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "./cache/transformers"
os.environ['MPLCONFIGDIR'] = './cache/matplotlib'
```

## Files Created

- `new_venv/`: The virtual environment directory
- `setup_venv.sh`: Shell script to activate venv and set cache dirs
- `activate_new_venv.sh`: Comprehensive setup script
- `setup_cache.py`: Python script to set up cache directories
- `VENV_README.md`: This documentation file

## Notes

- The cache directories are created automatically when the scripts are run
- All cached files are stored locally within the project directory
- This setup prevents conflicts with other projects or system-wide caches 