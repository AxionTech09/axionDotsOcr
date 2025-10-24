# CPU-Based Server Setup Guide for dots.ocr

This guide explains how to set up and run dots.ocr on a CPU-based server, addressing common issues like BFloat16/Float32 dtype mismatches.

## 🚨 Quick Fix for Your Current Error

The error you're experiencing is due to a dtype mismatch between BFloat16 and Float32 tensors. Here's the immediate fix:

### Step 1: Run Diagnostics
```bash
cd /var/www/dots_ocr
python diagnose_cpu.py
```

### Step 2: Convert Model to Float32
```bash
# Check current model dtypes
python convert_model_cpu.py /var/www/dots_ocr/dots_ocr/local_model --check-only

# Convert model to float32 (creates backup automatically)
python convert_model_cpu.py /var/www/dots_ocr/dots_ocr/local_model
```

### Step 3: Set Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export NUMEXPR_NUM_THREADS=$(nproc)
```

### Step 4: Test the Fix
```bash
python run_ocr.py your_test_image.jpg
```

## 🔧 Comprehensive Setup

### 1. System Requirements
- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 16GB+ (32GB recommended for large documents)
- **Python**: 3.9+
- **Operating System**: Linux (Ubuntu 20.04+), macOS, or Windows

### 2. Environment Setup

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### CPU Optimization
```bash
# Add to your shell profile (.bashrc, .zshrc, etc.)
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export NUMEXPR_NUM_THREADS=$(nproc)

# Reload your shell or run:
source ~/.bashrc
```

#### Python Configuration
```python
# Add to your script or create cpu_setup.py
import torch
import os

# Disable CUDA completely
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Configure PyTorch for CPU
torch.set_num_threads(os.cpu_count())
torch.set_default_dtype(torch.float32)

print("CPU setup completed")
```

### 3. Model Conversion (If Needed)

If you encounter BFloat16 errors, convert your model:

```bash
# Check if conversion is needed
python convert_model_cpu.py /path/to/your/model --check-only

# Convert model (creates backup)
python convert_model_cpu.py /path/to/your/model

# Or convert to a new location
python convert_model_cpu.py /path/to/your/model --output /path/to/cpu_model
```

### 4. Performance Optimization

#### Memory Management
```python
import gc
import torch

# Clean up memory regularly
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Use this after processing large documents
cleanup_memory()
```

#### CPU-Specific Settings
```python
# In your code, use these settings for better CPU performance
generation_kwargs = {
    "max_new_tokens": 2048,
    "do_sample": False,  # Deterministic for CPU
    "temperature": None,  # Disable for greedy decode
    "top_p": None,       # Disable for greedy decode
    "num_beams": 1,      # Single beam for speed
    "pad_token_id": tokenizer.eos_token_id
}
```

### 5. Troubleshooting Common Issues

#### Issue: "Input type (c10::BFloat16) and bias type (float) should be the same"
**Solution**: Convert model to float32
```bash
python convert_model_cpu.py /var/www/dots_ocr/dots_ocr/local_model
```

#### Issue: Out of Memory
**Solutions**:
1. Reduce image resolution
2. Process documents page by page
3. Increase system swap space
4. Use model quantization

#### Issue: Slow Performance
**Solutions**:
1. Set correct thread counts
2. Use CPU-optimized PyTorch build
3. Enable MKL-DNN if available
4. Process smaller batches

#### Issue: CUDA Errors on CPU-Only System
**Solution**: 
```bash
export CUDA_VISIBLE_DEVICES=""
```

### 6. Production Deployment

#### Docker Configuration
```dockerfile
FROM python:3.9-slim

# CPU optimization
ENV CUDA_VISIBLE_DEVICES=""
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8
ENV NUMEXPR_NUM_THREADS=8

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Convert model for CPU if needed
RUN python convert_model_cpu.py ./dots_ocr/local_model --check-only || \
    python convert_model_cpu.py ./dots_ocr/local_model

CMD ["python", "run_ocr.py"]
```

#### Systemd Service
```ini
[Unit]
Description=dots.ocr CPU Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/dots_ocr
Environment=CUDA_VISIBLE_DEVICES=""
Environment=OMP_NUM_THREADS=8
Environment=MKL_NUM_THREADS=8
ExecStart=/var/www/dots_ocr/venv/bin/python run_ocr.py
Restart=always

[Install]
WantedBy=multi-user.target
```

### 7. Monitoring and Optimization

#### Resource Monitoring
```bash
# Monitor during processing
htop
iotop
nvidia-smi  # Should show no processes if CUDA disabled
```

#### Performance Testing
```bash
# Test with sample document
time python run_ocr.py sample.pdf

# Monitor memory usage
python -c "
from dots_ocr.utils.cpu_optimizer import CPUOptimizer
import time
while True:
    info = CPUOptimizer.get_memory_info()
    print(f'Memory: {info[\"cpu_memory_used_gb\"]:.1f}GB/{info[\"cpu_memory_total_gb\"]:.1f}GB')
    time.sleep(5)
"
```

### 8. Configuration Files

#### Create `cpu_config.py`
```python
"""CPU-specific configuration for dots.ocr"""
import os
import torch

def setup_cpu_environment():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
    os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
    torch.set_num_threads(os.cpu_count())
    torch.set_default_dtype(torch.float32)

# Call this at the start of your application
setup_cpu_environment()
```

## 🚀 Quick Start Script

Save this as `setup_cpu.sh` and run with `bash setup_cpu.sh`:

```bash
#!/bin/bash
set -e

echo "Setting up dots.ocr for CPU inference..."

# Set environment variables
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)

# Install dependencies
pip install -r requirements.txt

# Run diagnostics
python diagnose_cpu.py

# Convert model if needed
if python convert_model_cpu.py ./dots_ocr/local_model --check-only; then
    echo "Model is already CPU-compatible"
else
    echo "Converting model for CPU..."
    python convert_model_cpu.py ./dots_ocr/local_model
fi

# Test with a sample
echo "Setup complete! Test with:"
echo "python run_ocr.py your_image.jpg"
```

## 📞 Support

If you continue to have issues:

1. Run `python diagnose_cpu.py` and share the output
2. Check the model conversion with `python convert_model_cpu.py /path/to/model --check-only`
3. Verify environment variables are set correctly
4. Monitor resource usage during inference

The key to successful CPU inference is ensuring consistent float32 datatypes throughout the model and inputs.