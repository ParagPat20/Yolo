#!/usr/bin/env python3
"""
GPU Diagnostic Script
Check CUDA and PyTorch GPU availability
"""

import sys

def check_gpu_status():
    print("🔧 GPU Diagnostic Tool")
    print("=" * 50)
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ PyTorch CUDA compiled: {torch.version.cuda}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✓ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
        else:
            print("❌ CUDA not available in PyTorch")
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    # Check NVIDIA driver
    print("\n🖥️  NVIDIA Driver Check:")
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ NVIDIA driver installed")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version' in line:
                    print(f"  {line.strip()}")
                if 'RTX 4060' in line or '4060' in line:
                    print(f"  Found: {line.strip()}")
        else:
            print("❌ nvidia-smi failed")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ nvidia-smi not found - NVIDIA drivers may not be installed")
    
    # Check CUDA toolkit
    print("\n🛠️  CUDA Toolkit Check:")
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ CUDA toolkit installed")
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    print(f"  {line.strip()}")
        else:
            print("⚠️  CUDA toolkit not found (not required for PyTorch)")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("⚠️  nvcc not found (CUDA toolkit not installed)")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if not torch.cuda.is_available():
        print("1. Reinstall PyTorch with CUDA support:")
        print("   pip uninstall torch torchvision torchaudio")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("\n2. Or for CUDA 11.8:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("\n3. Restart your terminal/IDE after installation")
    else:
        print("✓ GPU setup looks good!")
    
    return torch.cuda.is_available()

if __name__ == "__main__":
    check_gpu_status()
