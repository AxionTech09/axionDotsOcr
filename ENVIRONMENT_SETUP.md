# Environment Setup for dots.ocr

This document explains how to set up environment variables for the dots.ocr project, specifically the Hugging Face token required for model access.

## Quick Setup

### Method 1: Interactive Setup (Recommended)
Run the setup script to configure your environment interactively:

```bash
python setup_env.py
```

This script will:
- Guide you through getting a Hugging Face token
- Save it securely to your `.env` file
- Verify the setup works correctly

### Method 2: Manual .env File Setup
1. Create a `.env` file in the project root (if it doesn't exist)
2. Add your Hugging Face token:

```env
# Environment variables for dots.ocr project
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

3. Replace `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` with your actual token

### Method 3: PowerShell Environment Variable
For the current session only:

```powershell
$env:HF_TOKEN='hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
```

For permanent setup in PowerShell profile:
```powershell
# Add to your PowerShell profile
[Environment]::SetEnvironmentVariable('HF_TOKEN', 'hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx', 'User')
```

### Method 4: Command Prompt
For the current session only:
```cmd
set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Method 5: Windows System Environment Variables
1. Open Windows Settings → System → About → Advanced system settings
2. Click "Environment Variables"
3. Under "User variables", click "New"
4. Variable name: `HF_TOKEN`
5. Variable value: Your Hugging Face token
6. Click OK and restart your terminal

## Getting Your Hugging Face Token

1. Go to [Hugging Face Settings - Tokens](https://huggingface.co/settings/tokens)
2. Click "Create new token" or use an existing one
3. Give it a name (e.g., "dots-ocr")
4. Select "Read" access type
5. Click "Generate a token"
6. Copy the token (starts with `hf_`)

## Verification

To verify your environment setup:

```python
# Test if environment is set up correctly
from dots_ocr.utils.env_config import setup_environment

if setup_environment():
    print("✅ Environment setup successful!")
else:
    print("❌ Environment setup failed")
```

Or run:
```bash
python -c "from dots_ocr.utils.env_config import setup_environment; setup_environment()"
```

## Security Notes

- **Never commit your `.env` file to git** (it's already in `.gitignore`)
- **Never share your Hugging Face token** in code, issues, or public forums
- **Use environment variables** instead of hardcoding tokens in your code
- **Regenerate tokens** if you suspect they've been compromised

## Troubleshooting

### "HF_TOKEN not found" Error
- Make sure you've set the token using one of the methods above
- Restart your terminal/IDE after setting system environment variables
- Check that your `.env` file is in the correct location (project root)
- Verify the token format starts with `hf_`

### Authentication Errors
- Verify your token is valid by visiting [Hugging Face](https://huggingface.co/settings/tokens)
- Make sure your token has "Read" permissions
- Check if you need to accept any model license agreements on Hugging Face

### .env File Not Loading
- Ensure the `.env` file is in the project root directory
- Check file encoding is UTF-8
- Verify there are no extra spaces around the `=` sign
- Make sure the project imports the environment configuration

## File Structure

After setup, your project should have:

```
dots.ocr/
├── .env                     # Your environment variables (not in git)
├── .gitignore              # Includes .env to prevent committing tokens
├── setup_env.py            # Interactive setup script
├── dots_ocr/
│   └── utils/
│       └── env_config.py   # Environment configuration utilities
└── ...
```

## Integration with Code

The environment configuration is automatically loaded when you import dots_ocr modules:

```python
from dots_ocr.parser import DotsOCR  # Automatically loads environment
from dots_ocr.utils.env_config import get_hf_token  # Manual token access

# Your token is now available
ocr = DotsOCR()  # Uses HF_TOKEN automatically
```