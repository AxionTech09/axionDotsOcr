#!/usr/bin/env python3
"""
Model conversion utility to fix BFloat16/Float32 dtype mismatches for CPU inference.

This script converts a model from mixed precision (BFloat16/Float32) to pure Float32
for CPU compatibility.
"""
import os
import sys
import torch
import argparse
from pathlib import Path


def convert_model_to_float32(model_path, output_path=None, backup=True):
    """
    Convert a model from mixed precision to float32 for CPU inference.
    
    Args:
        model_path: Path to the model directory
        output_path: Path to save the converted model (optional)
        backup: Whether to create a backup of the original model
    """
    print(f"🔧 Converting model to float32: {model_path}")
    
    if output_path is None:
        output_path = model_path
    
    # Create backup if requested
    if backup and model_path == output_path:
        backup_path = f"{model_path}_backup"
        if not os.path.exists(backup_path):
            print(f"📦 Creating backup: {backup_path}")
            import shutil
            shutil.copytree(model_path, backup_path)
    
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
        
        # Load model configuration
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"✅ Loaded config for: {config.model_type}")
        
        # Load model with explicit float32
        print("📥 Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        # Convert all parameters to float32
        print("🔄 Converting to float32...")
        model = model.float()
        
        # Recursively ensure all parameters are float32
        def ensure_float32(module):
            for child in module.children():
                ensure_float32(child)
            
            # Convert parameters
            for param_name, param in module.named_parameters(recurse=False):
                if param.dtype != torch.float32 and param.dtype.is_floating_point:
                    param.data = param.data.to(torch.float32)
                    print(f"  Converted parameter: {param_name} -> float32")
            
            # Convert buffers
            for buffer_name, buffer in module.named_buffers(recurse=False):
                if (buffer.dtype != torch.float32 and 
                    buffer.dtype.is_floating_point and 
                    buffer.dtype not in [torch.long, torch.int, torch.bool]):
                    buffer.data = buffer.data.to(torch.float32)
                    print(f"  Converted buffer: {buffer_name} -> float32")
        
        ensure_float32(model)
        
        # Update config to specify float32
        config.torch_dtype = torch.float32
        
        # Save converted model
        print(f"💾 Saving converted model to: {output_path}")
        model.save_pretrained(
            output_path,
            safe_serialization=True,  # Use safetensors
            max_shard_size="5GB"
        )
        
        # Save updated config
        config.save_pretrained(output_path)
        
        print("✅ Model conversion completed successfully!")
        
        # Verify conversion
        print("🔍 Verifying conversion...")
        verify_model = AutoModelForCausalLM.from_pretrained(
            output_path,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        
        # Check all parameters are float32
        all_float32 = True
        for name, param in verify_model.named_parameters():
            if param.dtype != torch.float32 and param.dtype.is_floating_point:
                print(f"⚠️  Parameter {name} is still {param.dtype}")
                all_float32 = False
        
        if all_float32:
            print("✅ Verification passed: All parameters are float32")
        else:
            print("❌ Verification failed: Some parameters are not float32")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False


def check_model_dtypes(model_path):
    """Check the data types used in a model."""
    print(f"🔍 Checking model dtypes: {model_path}")
    
    try:
        from transformers import AutoModelForCausalLM
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        dtype_counts = {}
        for name, param in model.named_parameters():
            dtype = str(param.dtype)
            if dtype not in dtype_counts:
                dtype_counts[dtype] = 0
            dtype_counts[dtype] += 1
        
        print("📊 Parameter dtypes:")
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} parameters")
        
        # Check for mixed precision
        if len(dtype_counts) > 1:
            print("⚠️  Mixed precision detected - conversion recommended for CPU")
            return False
        elif 'torch.bfloat16' in dtype_counts:
            print("⚠️  BFloat16 detected - conversion required for CPU")
            return False
        elif 'torch.float32' in dtype_counts:
            print("✅ Float32 detected - should work on CPU")
            return True
        else:
            print("❓ Unknown dtype configuration")
            return False
        
    except Exception as e:
        print(f"❌ Failed to check model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert model to float32 for CPU inference")
    parser.add_argument("model_path", help="Path to model directory")
    parser.add_argument("--output", "-o", help="Output path (default: overwrite input)")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backup")
    parser.add_argument("--check-only", action="store_true", help="Only check dtypes, don't convert")
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model path does not exist: {model_path}")
        return 1
    
    if args.check_only:
        success = check_model_dtypes(model_path)
        return 0 if success else 1
    
    output_path = args.output if args.output else str(model_path)
    backup = not args.no_backup
    
    success = convert_model_to_float32(
        str(model_path), 
        output_path, 
        backup=backup
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())