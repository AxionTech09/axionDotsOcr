"""
Setup script for dots.ocr environment configuration.
Run this script to configure your environment variables.
"""
import os
import sys
from pathlib import Path


def setup_hf_token():
    """Interactive setup for Hugging Face token."""
    print("🔧 Setting up Hugging Face Token for dots.ocr")
    print("="*50)
    
    # Check if token already exists
    existing_token = os.environ.get("HF_TOKEN")
    if existing_token:
        print(f"✅ HF_TOKEN already set (length: {len(existing_token)} characters)")
        choice = input("Do you want to update it? (y/N): ").strip().lower()
        if choice not in ['y', 'yes']:
            return
    
    print("\n📝 To get your Hugging Face token:")
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Create a new token or copy an existing one")
    print("3. Make sure it has 'Read' access to repositories")
    print("\n⚠️  Note: Your token will be saved to .env file (not committed to git)")
    
    while True:
        token = input("\n🔑 Enter your Hugging Face token: ").strip()
        
        if not token:
            print("❌ Token cannot be empty!")
            continue
        
        if not token.startswith('hf_'):
            print("⚠️  Warning: Hugging Face tokens usually start with 'hf_'")
            confirm = input("Continue anyway? (y/N): ").strip().lower()
            if confirm not in ['y', 'yes']:
                continue
        
        break
    
    # Save to .env file
    env_path = Path(__file__).parent / '.env'
    
    try:
        # Read existing .env content
        existing_lines = []
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                existing_lines = f.readlines()
        
        # Update or add HF_TOKEN
        token_found = False
        updated_lines = []
        
        for line in existing_lines:
            if line.strip().startswith('HF_TOKEN='):
                updated_lines.append(f'HF_TOKEN={token}\n')
                token_found = True
            else:
                updated_lines.append(line)
        
        if not token_found:
            updated_lines.append(f'HF_TOKEN={token}\n')
        
        # Write back to .env
        with open(env_path, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)
        
        print(f"✅ HF_TOKEN saved to {env_path}")
        
        # Set in current environment
        os.environ['HF_TOKEN'] = token
        print("✅ HF_TOKEN set in current session")
        
    except Exception as e:
        print(f"❌ Error saving token: {e}")
        return False
    
    return True


def verify_setup():
    """Verify the environment setup."""
    print("\n🔍 Verifying setup...")
    
    # Load environment
    from dots_ocr.utils.env_config import setup_environment
    
    if setup_environment():
        print("✅ Environment setup completed successfully!")
        print("\n🚀 You can now run dots.ocr with your Hugging Face token.")
    else:
        print("❌ Environment setup failed. Please check the errors above.")


def main():
    print("🎯 dots.ocr Environment Setup")
    print("="*40)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    if setup_hf_token():
        verify_setup()
    else:
        print("❌ Setup failed. Please try again.")
        sys.exit(1)
    
    print("\n📖 Additional setup methods:")
    print("1. Manual .env file: Edit .env and add HF_TOKEN=your_token")
    print("2. PowerShell: $env:HF_TOKEN='your_token'")
    print("3. CMD: set HF_TOKEN=your_token")
    print("4. System Environment Variables (Windows Settings)")


if __name__ == "__main__":
    main()