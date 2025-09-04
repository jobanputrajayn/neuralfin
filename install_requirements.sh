#!/bin/bash

# Automatic requirements installer that detects GPU availability
# and installs the appropriate JAX version (CUDA12 for GPU, regular for CPU).

set -e

echo "🔍 Detecting system configuration..."

# Check for GPU availability
GPU_AVAILABLE=false

# Check for NVIDIA GPU using nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    # Check if nvidia-smi lists any GPUs
    if nvidia-smi | grep -q "NVIDIA-SMI"; then
        if nvidia-smi -L | grep -q "GPU"; then
            echo "✅ NVIDIA GPU detected"
            GPU_AVAILABLE=true
        else
            echo "⚠️ NVIDIA libraries found, but no GPU detected"
        fi
    fi
fi

# Check for CUDA installation
if command -v nvcc &> /dev/null; then
    echo "✅ CUDA installation detected (but this does NOT mean a GPU is available)"
    # Do NOT set GPU_AVAILABLE=true here! CUDA toolkit may be present without a GPU.
fi

# Check for TPU environment (common in Google Colab or GCP)
TPU_AVAILABLE=false

# Robust TPU detection
if [ ! -z "$COLAB_TPU_ADDR" ] || [ ! -z "$TPU_NAME" ] || [ "$JAX_PLATFORM_NAME" = "tpu" ] || [ -f "/usr/lib/libtpu.so" ]; then
    echo "✅ TPU environment detected (COLAB_TPU_ADDR, TPU_NAME, JAX_PLATFORM_NAME, or /usr/lib/libtpu.so present)"
    TPU_AVAILABLE=true
else
    echo "No TPU environment variables or libtpu.so detected."
fi

# If no GPU or TPU detected, use CPU version
if [ "$GPU_AVAILABLE" = false ] && [ "$TPU_AVAILABLE" = false ]; then
    echo "❌ No GPU/TPU/CUDA detected, using CPU version"
fi

echo "\n=================================================="

# Install appropriate JAX variant FIRST
if [ "$TPU_AVAILABLE" = true ]; then
    echo "🚀 Installing JAX for TPU..."
    pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    if [ $? -eq 0 ]; then
        echo "✅ Successfully installed JAX for TPU"
    else
        echo "❌ Error installing JAX for TPU"
        exit 1
    fi
elif [ "$GPU_AVAILABLE" = true ]; then
    echo "🚀 Installing JAX for CUDA12 (GPU)..."
    pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    if [ $? -eq 0 ]; then
        echo "✅ Successfully installed JAX for CUDA12 (GPU)"
    else
        echo "❌ Error installing JAX for CUDA12 (GPU)"
        exit 1
    fi
else
    echo "🚀 Installing JAX for CPU..."
    pip install jax
    if [ $? -eq 0 ]; then
        echo "✅ Successfully installed JAX for CPU"
    else
        echo "❌ Error installing JAX for CPU"
        exit 1
    fi
fi

# Install base requirements
if [ -f "requirements.txt" ]; then
    echo "🚀 Installing base requirements from requirements.txt..."
    pip install -r requirements.txt
    echo "✅ Successfully installed base requirements"
else
    echo "❌ Error: requirements.txt not found!"
    exit 1
fi

echo ""
echo "🎉 Installation completed successfully!"
if [ "$TPU_AVAILABLE" = true ]; then
    echo "💡 You can now use JAX with TPU acceleration"
elif [ "$GPU_AVAILABLE" = true ]; then
    echo "💡 You can now use JAX with CUDA12 acceleration"
else
    echo "💡 You can now use JAX on CPU"
fi 