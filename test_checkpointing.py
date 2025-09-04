#!/usr/bin/env python3
"""
Test script for checkpointing functionality.

This script tests the save and restore functionality using a real GPT model
to ensure everything works correctly with the updated Orbax API.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import jax
import jax.numpy as jnp
import optax
from flax import nnx

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.gpt_classifier import GPTClassifier
from training.checkpointing import (
    save_checkpoint, 
    restore_checkpoint, 
    save_model_checkpoint, 
    restore_model_checkpoint
)


def create_test_model():
    """Create a test model for checkpointing."""
    model = GPTClassifier(
        num_classes=3,
        d_model=64,  # Smaller for testing
        num_heads=4,
        num_layers=2,
        d_ff=128,
        dropout_rate=0.1,
        input_features=5,  # Small number of features for testing
        num_tickers=5
    )
    return model


def test_basic_checkpointing():
    """Test basic save and restore functionality."""
    print("🧪 Testing basic checkpointing...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "test_checkpoints"
        
        # Create test model
        model = create_test_model()
        
        # Create some test data
        test_input = jnp.ones((1, 60, 5), dtype=jnp.float32)
        
        # Test model forward pass
        try:
            output = model(test_input)
            print(f"✅ Model forward pass successful, output shape: {output.shape}")
        except Exception as e:
            print(f"❌ Model forward pass failed: {e}")
            return False
        
        # Test scalers
        scaler_mean = 100.0
        scaler_std = 50.0
        step = 42
        
        # Save checkpoint
        try:
            saved_path = save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=model,
                opt_state=None,
                scaler_mean=scaler_mean,
                scaler_std=scaler_std,
                step=step,
                learning_rate=1e-3,
                max_to_keep=2
            )
            print(f"✅ Checkpoint saved to: {saved_path}")
        except Exception as e:
            print(f"❌ Checkpoint save failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Verify checkpoint directory exists
        if not checkpoint_dir.exists():
            print(f"❌ Checkpoint directory not created: {checkpoint_dir}")
            return False
        
        # List checkpoint contents
        print(f"📁 Checkpoint directory contents:")
        for item in checkpoint_dir.rglob("*"):
            if item.is_file():
                print(f"   {item.relative_to(checkpoint_dir)}")
        
        # Restore checkpoint
        try:
            model_kwargs = {
                'num_classes': 3,
                'd_model': 64,
                'num_heads': 4,
                'num_layers': 2,
                'd_ff': 128,
                'dropout_rate': 0.1,
                'input_features': 5,
                'num_tickers': 5
            }
            
            params, state, restored_scaler_mean, restored_scaler_std, restored_step, restored_lr, opt_state = restore_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model_class=GPTClassifier,
                model_kwargs=model_kwargs,
                step=step,
                learning_rate=1e-3,
                create_optimizer_fn=lambda m, lr: nnx.Optimizer(m, optax.adam(lr))
            )
            
            if params is None:
                print("❌ Checkpoint restore failed - params is None")
                return False
            
            print(f"✅ Checkpoint restored successfully")
            print(f"   Restored scaler_mean: {restored_scaler_mean}")
            print(f"   Restored scaler_std: {restored_scaler_std}")
            print(f"   Restored step: {restored_step}")
            
            # Verify scalers match
            if abs(restored_scaler_mean - scaler_mean) > 1e-6:
                print(f"❌ Scaler mean mismatch: {restored_scaler_mean} != {scaler_mean}")
                return False
            
            if abs(restored_scaler_std - scaler_std) > 1e-6:
                print(f"❌ Scaler std mismatch: {restored_scaler_std} != {scaler_std}")
                return False
            
            if restored_step != step:
                print(f"❌ Step mismatch: {restored_step} != {step}")
                return False
            
            # Recreate model from restored params
            if state is not None:
                restored_model = nnx.merge(params, state)
            else:
                base_model = GPTClassifier(**model_kwargs)
                restored_model = nnx.merge(params, base_model)
            
            # Test restored model
            restored_output = restored_model(test_input)
            print(f"✅ Restored model forward pass successful, output shape: {restored_output.shape}")
            
            # Compare outputs (should be similar)
            output_diff = jnp.abs(output - restored_output).max()
            print(f"   Max output difference: {output_diff}")
            
            if output_diff > 1e-3:
                print(f"❌ Output difference too large: {output_diff}")
                return False
            
        except Exception as e:
            print(f"❌ Checkpoint restore failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("✅ Basic checkpointing test passed!")
    return True


def main():
    """Run all checkpointing tests."""
    print("🚀 Starting checkpointing tests...")
    print("=" * 50)
    
    # Set JAX memory fraction for testing
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
    
    tests = [
        test_basic_checkpointing,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ Test {test.__name__} failed")
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Checkpointing functionality is working correctly.")
        return 0
    else:
        print("💥 Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())
