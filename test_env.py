"""
Test script to verify environment configuration works correctly.
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_env_config():
    """Test the environment configuration."""
    print("🧪 Testing Environment Configuration")
    print("="*40)
    
    try:
        from dots_ocr.utils.env_config import setup_environment, get_hf_token
        
        print("\n1. Testing environment setup...")
        success = setup_environment()
        
        print("\n2. Testing token retrieval...")
        token = get_hf_token()
        
        if token:
            print(f"✅ HF_TOKEN found (length: {len(token)} characters)")
            print(f"   Token preview: {token[:10]}...{token[-4:] if len(token) > 10 else ''}")
        else:
            print("❌ HF_TOKEN not found")
        
        print("\n3. Testing direct environment access...")
        direct_token = os.environ.get('HF_TOKEN')
        if direct_token:
            print("✅ HF_TOKEN accessible via os.environ")
        else:
            print("❌ HF_TOKEN not in os.environ")
        
        print("\n" + "="*40)
        if success and token:
            print("🎉 All tests passed! Environment is properly configured.")
            return True
        else:
            print("❌ Some tests failed. Please check your configuration.")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're running from the project root directory.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def show_setup_instructions():
    """Show setup instructions if environment is not configured."""
    print("\n📝 Setup Instructions:")
    print("-" * 20)
    print("1. Run the interactive setup:")
    print("   python setup_env.py")
    print("\n2. Or manually create .env file with:")
    print("   HF_TOKEN=your_hugging_face_token_here")
    print("\n3. Or set environment variable in PowerShell:")
    print("   $env:HF_TOKEN='your_token'")


if __name__ == "__main__":
    success = test_env_config()
    
    if not success:
        show_setup_instructions()
        sys.exit(1)
    else:
        print("\n🚀 You're ready to use dots.ocr!")