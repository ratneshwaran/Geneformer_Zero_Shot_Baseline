import os
import sys

def setup_cache_directories():
    """Set up cache directories locally within the project."""
    
    # Define cache directories relative to the project root
    cache_base = "./cache"
    cache_dirs = {
        "PIP_CACHE_DIR": f"{cache_base}/pip",
        "HF_HOME": f"{cache_base}/huggingface", 
        "TRANSFORMERS_CACHE": f"{cache_base}/transformers",
        "MPLCONFIGDIR": f"{cache_base}/matplotlib"
    }
    
    # Create directories and set environment variables
    for env_var, cache_path in cache_dirs.items():
        # Create directory if it doesn't exist
        os.makedirs(cache_path, exist_ok=True)
        
        # Set environment variable
        os.environ[env_var] = cache_path
        
        print(f"Set {env_var} = {cache_path}")
    
    print("\nCache directories have been set up locally!")
    print("You can now install packages and they will be cached in the local ./cache directory.")

if __name__ == "__main__":
    setup_cache_directories() 