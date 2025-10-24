"""
Environment configuration utilities for dots.ocr project.
Handles loading environment variables from .env file and system environment.
"""
import os
from pathlib import Path
from typing import Optional


def load_env_file(env_path: Optional[str] = None) -> None:
    """
    Load environment variables from .env file.
    
    Args:
        env_path: Path to .env file. If None, looks for .env in project root.
    """
    if env_path is None:
        # Find project root (directory containing this file's parent)
        current_dir = Path(__file__).parent.parent
        env_path = current_dir / '.env'
    
    env_path = Path(env_path)
    
    if not env_path.exists():
        print(f"⚠️  No .env file found at {env_path}")
        return
    
    # Try using python-dotenv if available, fallback to manual parsing
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
        print(f"✅ Environment variables loaded from {env_path} (using python-dotenv)")
        return
    except ImportError:
        pass
    
    # Fallback to manual parsing
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse key=value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    
                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value
        
        print(f"✅ Environment variables loaded from {env_path} (manual parsing)")
        
    except Exception as e:
        print(f"❌ Error loading .env file: {e}")


def get_hf_token() -> Optional[str]:
    """
    Get Hugging Face token from environment variables.
    
    Returns:
        HF_TOKEN value if found, None otherwise.
    """
    # Try different common environment variable names
    token_names = ['HF_TOKEN', 'HUGGING_FACE_TOKEN', 'HUGGINGFACE_TOKEN']
    
    for token_name in token_names:
        token = os.environ.get(token_name)
        if token:
            return token
    
    return None


def setup_environment():
    """
    Setup environment for dots.ocr project.
    Loads .env file and validates required environment variables.
    """
    # Load .env file
    load_env_file()
    
    # Check for HF_TOKEN
    hf_token = get_hf_token()
    if not hf_token:
        print("⚠️  HF_TOKEN not found in environment variables.")
        print("   Please set your Hugging Face token in one of these ways:")
        print("   1. Add HF_TOKEN=your_token to .env file")
        print("   2. Set HF_TOKEN environment variable in your system")
        print("   3. Run: $env:HF_TOKEN='your_token' (PowerShell)")
        return False
    
    print(f"✅ HF_TOKEN found (length: {len(hf_token)} characters)")
    return True


if __name__ == "__main__":
    setup_environment()