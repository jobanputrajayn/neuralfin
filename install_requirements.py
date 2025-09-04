#!/usr/bin/env python3
"""
Automatic requirements installer that detects GPU availability
and installs the appropriate JAX version (CUDA12 for GPU, regular for CPU).
"""

import subprocess
import sys
import os
from pathlib import Path


def check_gpu_availability():
    """Check if GPU is available and CUDA is installed."""
    try:
        # Check for NVIDIA GPU using nvidia-smi
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        if result.returncode == 0 and 'GPU' in result.stdout:
            print("✅ NVIDIA GPU detected")
            return True
        elif result.returncode == 0:
            print("⚠️ NVIDIA libraries found, but no GPU detected")
    except FileNotFoundError:
        pass
    
    try:
        # Check for CUDA installation
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA installation detected (but no GPU detected)")
    except FileNotFoundError:
        pass
    
    print("❌ No GPU/CUDA detected, using CPU version")
    return False


def check_tpu_availability():
    """Robustly check if TPU environment is available."""
    tpu_env = False
    if os.environ.get("COLAB_TPU_ADDR"):
        print("✅ TPU detected via COLAB_TPU_ADDR")
        tpu_env = True
    if os.environ.get("PJRT_DEVICE", "").lower() == "tpu":
        print("✅ TPU detected via PJRT_DEVICE=TPU")
        tpu_env = True
    if os.environ.get("TPU_NAME"):
        print("✅ TPU detected via TPU_NAME")
        tpu_env = True
    if os.environ.get("JAX_PLATFORM_NAME", "").lower() == "tpu":
        print("✅ TPU detected via JAX_PLATFORM_NAME=tpu")
        tpu_env = True
    if os.path.exists("/usr/lib/libtpu.so"):
        print("✅ TPU detected via /usr/lib/libtpu.so presence")
        tpu_env = True
    if not tpu_env:
        print("No TPU environment variables or libtpu.so detected.")
    return tpu_env


def install_base_requirements():
    """Install all non-JAX requirements from requirements.txt."""
    requirements_file = "requirements.txt"
    print("🚀 Installing base requirements from requirements.txt...")
    if not Path(requirements_file).exists():
        print(f"❌ Error: {requirements_file} not found!")
        return False
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ], check=True)
        print(f"✅ Successfully installed base requirements from {requirements_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing base requirements: {e}")
        return False


def install_jax_variant(use_gpu=False, use_tpu=False):
    """Install the correct JAX variant for the detected hardware."""
    if use_tpu:
        print("🚀 Installing JAX for TPU...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "jax[tpu]", "-f", "https://storage.googleapis.com/jax-releases/libtpu_releases.html"
            ], check=True)
            print("✅ Successfully installed JAX for TPU")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing JAX for TPU: {e}")
            return False
    elif use_gpu:
        print("🚀 Installing JAX for CUDA12 (GPU)...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "jax[cuda12]", "-f", "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
            ], check=True)
            print("✅ Successfully installed JAX for CUDA12 (GPU)")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing JAX for CUDA12 (GPU): {e}")
            return False
    else:
        print("🚀 Installing JAX for CPU...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "jax"
            ], check=True)
            print("✅ Successfully installed JAX for CPU")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing JAX for CPU: {e}")
            return False


def main():
    """Main function to detect hardware and install appropriate requirements."""
    print("🔍 Detecting system configuration...")
    use_tpu = check_tpu_availability()
    use_gpu = False
    if not use_tpu:
        use_gpu = check_gpu_availability()

    print("\n" + "="*50)
    # Install JAX FIRST
    jax_success = install_jax_variant(use_gpu=use_gpu, use_tpu=use_tpu)
    if not jax_success:
        print("\n❌ Installation failed during JAX installation. Please check the error messages above.")
        sys.exit(1)

    base_success = install_base_requirements()
    if base_success:
        print("\n🎉 Installation completed successfully!")
        if use_tpu:
            print("💡 You can now use JAX with TPU acceleration")
        elif use_gpu:
            print("💡 You can now use JAX with CUDA12 acceleration")
        else:
            print("💡 You can now use JAX on CPU")
    else:
        print("\n❌ Installation failed during base requirements. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main() 