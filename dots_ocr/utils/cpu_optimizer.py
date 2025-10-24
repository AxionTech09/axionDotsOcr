"""
CPU optimization utilities for dots.ocr on CPU-based servers.
Handles memory management, dtype consistency, and performance optimization.
"""
import os
import torch
import gc
from typing import Optional


class CPUOptimizer:
    """Optimizes PyTorch models and tensors for CPU inference."""
    
    @staticmethod
    def setup_cpu_environment():
        """Configure environment for optimal CPU performance."""
        # Disable CUDA completely
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # Set CPU-specific environment variables
        os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
        os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
        os.environ["NUMEXPR_NUM_THREADS"] = str(os.cpu_count())
        
        # Set PyTorch to use all CPU cores
        torch.set_num_threads(os.cpu_count())
        
        # Force float32 as default
        torch.set_default_dtype(torch.float32)
        
        print(f"🔧 CPU environment configured:")
        print(f"   CPU cores: {os.cpu_count()}")
        print(f"   PyTorch threads: {torch.get_num_threads()}")
        print(f"   Default dtype: {torch.get_default_dtype()}")
    
    @staticmethod
    def optimize_model_for_cpu(model):
        """Convert model to optimal CPU configuration."""
        print("🔧 Optimizing model for CPU inference...")
        
        # Ensure model is in eval mode
        model.eval()
        
        # Convert to float32 recursively
        CPUOptimizer._convert_model_dtype(model, torch.float32)
        
        # Move to CPU
        model = model.to("cpu")
        
        # Enable CPU-specific optimizations
        if hasattr(torch.backends, 'mkldnn') and torch.backends.mkldnn.is_available():
            try:
                model = torch.jit.optimize_for_inference(model)
                print("✅ MKL-DNN optimization enabled")
            except:
                print("⚠️  MKL-DNN optimization failed, continuing without it")
        
        return model
    
    @staticmethod
    def _convert_model_dtype(model, target_dtype=torch.float32):
        """Recursively convert all model parameters to target dtype."""
        
        def convert_module(module):
            # Convert child modules first
            for child in module.children():
                convert_module(child)
            
            # Convert parameters
            for param_name, param in module.named_parameters(recurse=False):
                if param.dtype != target_dtype and param.dtype.is_floating_point:
                    param.data = param.data.to(target_dtype)
            
            # Convert buffers (but preserve integer and boolean types)
            for buffer_name, buffer in module.named_buffers(recurse=False):
                if (buffer.dtype != target_dtype and 
                    buffer.dtype.is_floating_point and 
                    buffer.dtype not in [torch.long, torch.int, torch.bool, torch.int32, torch.int64]):
                    buffer.data = buffer.data.to(target_dtype)
        
        convert_module(model)
    
    @staticmethod
    def prepare_inputs_for_cpu(inputs):
        """Prepare input tensors for CPU inference with proper dtypes."""
        cpu_inputs = {}
        
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                # Handle different tensor types appropriately
                if key in ["input_ids", "attention_mask"] or "token" in key.lower():
                    # Keep ID tensors as integers
                    cpu_inputs[key] = value.to(device="cpu")
                elif key in ["pixel_values", "image_grid_thw"] or "image" in key.lower():
                    # Convert image tensors to float32
                    cpu_inputs[key] = value.to(dtype=torch.float32, device="cpu")
                else:
                    # Default: convert to float32 if it's a floating point tensor
                    if value.dtype.is_floating_point:
                        cpu_inputs[key] = value.to(dtype=torch.float32, device="cpu")
                    else:
                        cpu_inputs[key] = value.to(device="cpu")
            else:
                cpu_inputs[key] = value
        
        return cpu_inputs
    
    @staticmethod
    def cleanup_memory():
        """Clean up GPU and CPU memory."""
        # Clear CUDA cache if CUDA is available (just in case)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
    
    @staticmethod
    def get_memory_info():
        """Get current memory usage information."""
        import psutil
        
        # CPU memory
        memory = psutil.virtual_memory()
        cpu_memory_used = memory.used / (1024**3)  # GB
        cpu_memory_total = memory.total / (1024**3)  # GB
        cpu_memory_percent = memory.percent
        
        info = {
            "cpu_memory_used_gb": round(cpu_memory_used, 2),
            "cpu_memory_total_gb": round(cpu_memory_total, 2),
            "cpu_memory_percent": cpu_memory_percent
        }
        
        # GPU memory (if available)
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info["gpu_memory_used_gb"] = round(gpu_memory_used, 2)
            info["gpu_memory_total_gb"] = round(gpu_memory_total, 2)
        
        return info


def configure_cpu_inference():
    """Main function to configure the environment for CPU inference."""
    print("🚀 Configuring environment for CPU-based inference...")
    
    # Setup CPU environment
    CPUOptimizer.setup_cpu_environment()
    
    # Disable any CUDA operations
    torch.backends.cudnn.enabled = False
    
    # Configure for CPU-only execution
    if hasattr(torch.backends, 'mps'):
        torch.backends.mps.enabled = False
    
    print("✅ CPU inference configuration completed")
    
    # Show memory info
    memory_info = CPUOptimizer.get_memory_info()
    print(f"💾 System memory: {memory_info['cpu_memory_used_gb']:.1f}GB / {memory_info['cpu_memory_total_gb']:.1f}GB ({memory_info['cpu_memory_percent']:.1f}%)")


if __name__ == "__main__":
    configure_cpu_inference()