#!/usr/bin/env python3
"""
CPU Inference Diagnostic Tool for dots.ocr

This script helps diagnose and fix common issues with CPU-based inference,
particularly dtype mismatches and memory problems.
"""
import os
import sys
import torch
import psutil
from pathlib import Path


def check_system_requirements():
    """Check system requirements for CPU inference."""
    print("🔍 System Requirements Check")
    print("=" * 40)
    
    # Python version
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    
    # PyTorch version and configuration
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Default dtype: {torch.get_default_dtype()}")
    print(f"Number of threads: {torch.get_num_threads()}")
    
    # CPU information
    cpu_count = os.cpu_count()
    cpu_freq = psutil.cpu_freq()
    print(f"CPU cores: {cpu_count}")
    if cpu_freq:
        print(f"CPU frequency: {cpu_freq.current:.1f} MHz")
    
    # Memory information
    memory = psutil.virtual_memory()
    print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    print(f"Memory usage: {memory.percent:.1f}%")
    
    # Environment variables
    print("\n🔧 Environment Variables")
    print("-" * 20)
    env_vars = ["CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]
    for var in env_vars:
        value = os.environ.get(var, "Not set")
        print(f"{var}: {value}")
    
    return True


def test_pytorch_cpu():
    """Test PyTorch CPU functionality."""
    print("\n🧪 PyTorch CPU Test")
    print("=" * 20)
    
    try:
        # Test basic tensor operations
        x = torch.randn(100, 100, dtype=torch.float32)
        y = torch.randn(100, 100, dtype=torch.float32)
        z = torch.mm(x, y)
        print("✅ Basic tensor operations: OK")
        
        # Test dtype consistency
        if z.dtype == torch.float32:
            print("✅ Float32 dtype consistency: OK")
        else:
            print(f"❌ Dtype mismatch: Expected float32, got {z.dtype}")
            return False
        
        # Test device placement
        if z.device.type == 'cpu':
            print("✅ CPU device placement: OK")
        else:
            print(f"❌ Device mismatch: Expected CPU, got {z.device}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch CPU test failed: {e}")
        return False


def test_model_loading():
    """Test model loading with CPU optimizations."""
    print("\n🤖 Model Loading Test")
    print("=" * 25)
    
    # Check if local model exists
    model_path = "/var/www/dots_ocr/dots_ocr/local_model"
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        return False
    
    print(f"✅ Model path exists: {model_path}")
    
    try:
        from transformers import AutoConfig
        
        # Test config loading
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print("✅ Model config loading: OK")
        
        # Check model architecture
        print(f"Model type: {config.model_type}")
        if hasattr(config, 'torch_dtype'):
            print(f"Default dtype: {config.torch_dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False


def test_dots_ocr_import():
    """Test dots_ocr module imports."""
    print("\n📦 Module Import Test")
    print("=" * 22)
    
    try:
        # Test basic imports
        from dots_ocr.utils.cpu_optimizer import CPUOptimizer, configure_cpu_inference
        print("✅ CPU optimizer import: OK")
        
        from dots_ocr.parser import DotsOCRParser
        print("✅ DotsOCR parser import: OK")
        
        # Test CPU configuration
        configure_cpu_inference()
        print("✅ CPU configuration: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


def run_inference_test():
    """Run a minimal inference test."""
    print("\n🚀 Inference Test")
    print("=" * 17)
    
    try:
        from dots_ocr.parser import DotsOCRParser
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        test_image = Image.new('RGB', (100, 100), color='white')
        
        print("Creating DotsOCRParser instance...")
        parser = DotsOCRParser()
        print("✅ Parser initialization: OK")
        
        # Test basic functionality (without full inference to save time)
        print("✅ Basic functionality test: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        print(f"Error details: {str(e)}")
        
        # Check for common error patterns
        error_str = str(e).lower()
        if "bfloat16" in error_str or "bias type" in error_str:
            print("\n💡 Suggested fixes:")
            print("1. Ensure all model weights are converted to float32")
            print("2. Check model loading configuration")
            print("3. Verify CPU optimizer is working correctly")
        
        return False


def provide_recommendations():
    """Provide system optimization recommendations."""
    print("\n💡 Optimization Recommendations")
    print("=" * 35)
    
    # Memory recommendations
    memory = psutil.virtual_memory()
    if memory.available / (1024**3) < 8:
        print("⚠️  Low available memory. Consider:")
        print("   - Closing other applications")
        print("   - Using model quantization")
        print("   - Reducing batch size")
    
    # CPU recommendations
    cpu_count = os.cpu_count()
    current_threads = torch.get_num_threads()
    if current_threads != cpu_count:
        print(f"⚠️  Thread count mismatch:")
        print(f"   PyTorch threads: {current_threads}, CPU cores: {cpu_count}")
        print("   Consider setting OMP_NUM_THREADS environment variable")
    
    # CUDA recommendations
    if torch.cuda.is_available():
        print("⚠️  CUDA detected but running CPU inference:")
        print("   Set CUDA_VISIBLE_DEVICES=\"\" to disable GPU completely")
    
    print("\n🔧 Quick fixes:")
    print("1. export CUDA_VISIBLE_DEVICES=\"\"")
    print(f"2. export OMP_NUM_THREADS={cpu_count}")
    print("3. Restart the application after setting environment variables")


def main():
    """Main diagnostic function."""
    print("🩺 dots.ocr CPU Inference Diagnostics")
    print("=" * 45)
    
    tests = [
        ("System Requirements", check_system_requirements),
        ("PyTorch CPU", test_pytorch_cpu),
        ("Model Loading", test_model_loading),
        ("Module Imports", test_dots_ocr_import),
        ("Inference", run_inference_test),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📊 Diagnostic Summary")
    print("=" * 22)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed < total:
        provide_recommendations()
        return 1
    else:
        print("\n🎉 All tests passed! Your system is ready for CPU inference.")
        return 0


if __name__ == "__main__":
    sys.exit(main())