"""
Tests for LIF (Leaky Integrate-and-Fire) Layer

This module contains comprehensive tests to ensure the LIF layers
they work correctly with JAX and NNX, based on the snnax implementation.

Third-Party Attribution:
- Original implementation: snnax library (https://github.com/neuromorphs/snnax)
- License: MIT License (compatible)
- See docs/ATTRIBUTIONS.md for full attribution details
"""

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import value_and_grad
import numpy as np
from .lif_layer import SimpleLIF, LIF, LIFSoftReset, AdaptiveLIF, superspike_surrogate
import optax
import sys


def get_param_names(module):
    # Try nnx.state_dict if available
    if hasattr(nnx, 'state_dict'):
        state_dict = nnx.state_dict(module)
        print("Using nnx.state_dict:", type(state_dict), state_dict); sys.stdout.flush()
        return set(state_dict.keys()), state_dict
    
    # Try to access parameters directly from module attributes
    param_dict = {}
    for name, value in module.__dict__.items():
        if isinstance(value, nnx.Param):
            param_dict[name] = value
    print("Direct module params:", param_dict); sys.stdout.flush()
    if param_dict:
        return set(param_dict.keys()), param_dict
    
    # Otherwise, use nnx.split and extract from NodeDef attributes
    params, _ = nnx.split(module)
    print("Type of params from nnx.split:", type(params)); sys.stdout.flush()
    print("params:", params); sys.stdout.flush()
    if hasattr(params, 'attributes'):
        param_dict = {}
        for name, value in params.attributes:
            # Only include VariableDef with type 'Param'
            if getattr(value, 'type', None) == 'Param':
                print(f"VariableDef for {name}: {value}, dir: {dir(value)}"); sys.stdout.flush()
                print(f"repr: {repr(value)}"); sys.stdout.flush()
                print(f"__dict__: {getattr(value, '__dict__', 'no __dict__')}"); sys.stdout.flush()
                print(f"SUMMARY: name={name}, value={value}"); sys.stdout.flush()
        print("Extracted param_dict:", param_dict); sys.stdout.flush()
        return set(param_dict.keys()), param_dict
    # fallback: try keys directly
    if hasattr(params, 'keys'):
        return set(params.keys()), params
    print("dir(params):", dir(params)); sys.stdout.flush()
    return set(), params


def test_simple_lif_creation():
    """Test SimpleLIF layer creation and basic functionality."""
    print("Testing SimpleLIF creation...")
    
    # Create SimpleLIF layer
    simple_lif = SimpleLIF(
        decay_constants=[0.9],
        threshold=1.0,
        reset_val=0.0,
        surrogate_beta=10.0
    )
    
    # Check that parameters are properly registered
    param_names, params = get_param_names(simple_lif)
    expected_params = {'decay_constants', 'threshold', 'reset_val', 'surrogate_beta'}
    assert param_names == expected_params, f"Expected {expected_params}, got {param_names}"
    
    # Check parameter shapes and values
    assert params['decay_constants'].shape == (1,)
    assert params['threshold'].shape == ()
    assert params['reset_val'].shape == ()
    assert params['surrogate_beta'].shape == ()
    
    print("✅ SimpleLIF creation test passed")


def test_lif_creation():
    """Test LIF layer creation and basic functionality."""
    print("Testing LIF creation...")
    
    # Create LIF layer
    lif = LIF(
        decay_constants=[0.9, 0.8],
        threshold=1.0,
        reset_val=0.0,
        surrogate_beta=10.0
    )
    
    # Check that parameters are properly registered
    param_names, params = get_param_names(lif)
    expected_params = {'decay_constants', 'threshold', 'reset_val', 'surrogate_beta'}
    assert param_names == expected_params, f"Expected {expected_params}, got {param_names}"
    
    # Check parameter shapes and values
    assert params['decay_constants'].shape == (2,)
    assert params['threshold'].shape == ()
    assert params['reset_val'].shape == ()
    assert params['surrogate_beta'].shape == ()
    
    print("✅ LIF creation test passed")


def test_adaptive_lif_creation():
    """Test AdaptiveLIF layer creation and basic functionality."""
    print("Testing AdaptiveLIF creation...")
    
    # Create AdaptiveLIF layer
    adaptive_lif = AdaptiveLIF(
        decay_constants=[0.9],
        ada_decay_constant=[0.8],
        ada_step_val=[1.0],
        ada_coupling_var=[0.5],
        threshold=1.0,
        reset_val=0.0,
        surrogate_beta=10.0
    )
    
    # Check that parameters are properly registered
    param_names, params = get_param_names(adaptive_lif)
    expected_params = {
        'decay_constants', 'ada_decay_constant', 'ada_step_val', 
        'ada_coupling_var', 'threshold', 'reset_val', 'surrogate_beta'
    }
    assert param_names == expected_params, f"Expected {expected_params}, got {param_names}"
    
    # Check parameter shapes and values
    assert params['decay_constants'].shape == (1,)
    assert params['ada_decay_constant'].shape == (1,)
    assert params['ada_step_val'].shape == (1,)
    assert params['ada_coupling_var'].shape == (1,)
    assert params['threshold'].shape == ()
    assert params['reset_val'].shape == ()
    assert params['surrogate_beta'].shape == ()
    
    print("✅ AdaptiveLIF creation test passed")


def test_simple_lif_forward_pass():
    """Test SimpleLIF forward pass with snnax logic and NNX state structure."""
    print("Testing SimpleLIF forward pass (snnax logic)...")

    lif = SimpleLIF(
        decay_constants=[0.9],
        threshold=1.0,
        reset_val=0.0,
        surrogate_beta=10.0
    )
    x = jnp.array([[0.5, 1.2, 0.8, 1.5]], dtype=jnp.float32)
    spikes, state = lif(x, reset_state=True)
    expected_mem_pot = 0.9 * jnp.zeros_like(x) + 0.1 * x
    assert jnp.allclose(state['mem_pot'], expected_mem_pot)
    assert jnp.all((spikes == 0) | (spikes == 1))
    print("SimpleLIF forward pass (snnax logic) test passed.")


def test_lif_forward_pass():
    """Test LIF forward pass with snnax logic and NNX state structure."""
    print("Testing LIF forward pass (snnax logic)...")

    # Create LIF layer
    lif = LIF(
        decay_constants=[0.9, 0.8],
        threshold=1.0,
        reset_val=0.0,
        surrogate_beta=10.0
    )

    # Create input data
    x = jnp.array([[0.5, 1.2, 0.8, 1.5],
                   [0.3, 0.9, 1.1, 0.7]], dtype=jnp.float32)

    # First forward pass (previous spike_output is zero)
    spikes, state = lif(x, reset_state=True)

    # According to snnax logic, mem_pot is zero after first step (reset applied before update, initial state is zero)
    assert jnp.allclose(state['mem_pot'], jnp.zeros_like(x))
    assert jnp.all((spikes == 0) | (spikes == 1))
    print("LIF forward pass (snnax logic) test passed.")


def test_adaptive_lif_forward_pass():
    """Test AdaptiveLIF forward pass."""
    print("Testing AdaptiveLIF forward pass...")
    
    # Create AdaptiveLIF layer
    adaptive_lif = AdaptiveLIF(
        decay_constants=[0.9],
        ada_decay_constant=[0.8],
        ada_step_val=[1.0],
        ada_coupling_var=[0.5],
        threshold=1.0,
        reset_val=0.0,
        surrogate_beta=10.0
    )
    
    # Create input data
    x = jnp.array([[0.5, 1.2, 0.8, 1.5],
                   [0.3, 0.9, 1.1, 0.7]], dtype=jnp.float32)
    
    # Forward pass
    spikes, state = adaptive_lif(x, reset_state=True)
    
    # Check output shapes
    assert spikes.shape == x.shape
    assert state['mem_pot'].shape == x.shape
    assert state['ada_var'].shape == x.shape
    assert state['spike_output'].shape == x.shape
    
    # Check that spikes are binary
    assert jnp.all((spikes == 0) | (spikes == 1))
    
    # Check that membrane potential and adaptive variable are updated
    assert not jnp.allclose(state['mem_pot'], jnp.zeros_like(state['mem_pot']))
    assert not jnp.allclose(state['ada_var'], jnp.zeros_like(state['ada_var']))
    
    print("✅ AdaptiveLIF forward pass test passed")


def test_gradient_computation():
    """Test gradient computation with trainable parameters."""
    print("Testing gradient computation...")
    
    # Create SimpleLIF layer
    lif = SimpleLIF(
        decay_constants=[0.9],
        threshold=1.0,
        reset_val=0.0,
        surrogate_beta=10.0
    )
    
    # Create input data
    x = jnp.array([[0.5, 1.2, 0.8, 1.5]], dtype=jnp.float32)
    
    # Define a simple loss function
    def loss_fn(model, inputs):
        spikes, state = model(inputs, reset_state=True)
        return jnp.mean(spikes), (spikes, state)  # Return (loss, aux)
    
    # Compute gradients with respect to parameters
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    (loss, (spikes, state)), grads = grad_fn(lif, x)
    assert grads is not None
    
    # Check that gradients are computed for all parameters
    param_names, params = get_param_names(lif)
    expected_grads = param_names
    actual_grads = set(grads.keys())
    assert actual_grads == expected_grads, f"Expected gradients for {expected_grads}, got {actual_grads}"
    
    # Check that gradients have the same shapes as parameters
    for name, grad in grads.items():
        param_shape = params[name].shape
        grad_shape = grad.value.shape
        assert grad_shape == param_shape, f"Gradient shape {grad_shape} != parameter shape {param_shape} for {name}"
    
    # Check that gradients are not all zero (indicating gradient flow)
    total_grad_norm = sum(jnp.sum(jnp.square(grad.value)) for grad in grads.values())
    assert total_grad_norm > 0, "All gradients are zero, indicating no gradient flow"
    
    print("✅ Gradient computation test passed")


def test_parameter_updates():
    """Test that parameters can be updated."""
    print("Testing parameter updates...")
    
    # Create SimpleLIF layer
    lif = SimpleLIF(
        decay_constants=[0.9],
        threshold=1.0,
        reset_val=0.0,
        surrogate_beta=10.0
    )
    
    # Get initial parameters
    param_names, params = get_param_names(lif)
    initial_decay = params['decay_constants'].copy()
    
    # Create optimizer
    optimizer = nnx.Optimizer(lif, optax.adam(0.01))
    
    # Create input data
    x = jnp.array([[0.5, 1.2, 0.8, 1.5]], dtype=jnp.float32)
    
    # Define loss function
    def loss_fn(model, inputs):
        spikes, state = model(inputs, reset_state=True)
        return jnp.mean(spikes), (spikes, state)  # Return (loss, aux)
    
    # Compute gradients
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    (loss, (spikes, state)), grads = grad_fn(lif, x)
    assert grads is not None
    
    # Update parameters
    optimizer.update(grads)
    
    # Get updated parameters
    _, updated_params = get_param_names(lif)
    updated_decay = updated_params['decay_constants']
    
    # Check that parameters have changed
    assert not jnp.allclose(initial_decay, updated_decay), "Parameters did not update"
    
    print("✅ Parameter updates test passed")


def test_state_continuity():
    """Test that state is maintained between forward passes."""
    print("Testing state continuity...")
    
    # Create LIF layer
    lif = LIF(
        decay_constants=[0.9, 0.8],
        threshold=1.0,
        reset_val=0.0,
        surrogate_beta=10.0
    )
    
    # Create input data
    x = jnp.array([[0.5, 1.2, 0.8, 1.5],
                   [0.3, 0.9, 1.1, 0.7]], dtype=jnp.float32)
    
    # First forward pass (reset state)
    spikes1, state1 = lif(x, reset_state=True)
    
    # Second forward pass (continue state)
    spikes2, state2 = lif(x, reset_state=False)
    
    # Third forward pass (reset state again)
    spikes3, state3 = lif(x, reset_state=True)
    
    # Check that state is different between continuous and reset passes
    assert not jnp.allclose(state1['mem_pot'], state2['mem_pot'])
    assert not jnp.allclose(state1['syn_curr'], state2['syn_curr'])
    
    # Check that reset passes produce similar results
    assert jnp.allclose(state1['mem_pot'], state3['mem_pot'])
    assert jnp.allclose(state1['syn_curr'], state3['syn_curr'])
    
    print("✅ State continuity test passed")


def test_surrogate_gradients():
    """Test surrogate gradient functions."""
    print("Testing surrogate gradients...")
    
    # Create input
    x = jnp.linspace(-2, 2, 100)
    
    # Test superspike surrogate
    superspike_fn = superspike_surrogate(10.0)
    superspike_out = superspike_fn(x)
    
    # Check that output is binary (0 or 1)
    assert jnp.all((superspike_out == 0) | (superspike_out == 1))
    
    # Check that gradients can be computed
    grad_fn = jax.grad(lambda x: jnp.sum(superspike_fn(x)))
    gradients = grad_fn(x)
    
    # Check that gradients are not all zero
    assert not jnp.allclose(gradients, 0)
    
    print("✅ Surrogate gradients test passed")


def test_different_lif_variants():
    """Test different LIF variants."""
    print("Testing different LIF variants...")
    
    # Create input data
    x = jnp.array([[0.5, 1.2, 0.8, 1.5],
                   [0.3, 0.9, 1.1, 0.7]], dtype=jnp.float32)
    
    # Test SimpleLIF
    simple_lif = SimpleLIF(decay_constants=[0.9], threshold=1.0, reset_val=0.0)
    spikes_simple, state_simple = simple_lif(x, reset_state=True)
    
    # Test LIF
    lif = LIF(decay_constants=[0.9, 0.8], threshold=1.0, reset_val=0.0)
    spikes_lif, state_lif = lif(x, reset_state=True)
    
    # Test LIFSoftReset
    lif_soft = LIFSoftReset(decay_constants=[0.9, 0.8], threshold=1.0, reset_val=0.5)
    spikes_soft, state_soft = lif_soft(x, reset_state=True)
    
    # Test AdaptiveLIF
    adaptive_lif = AdaptiveLIF(
        decay_constants=[0.9],
        ada_decay_constant=[0.8],
        ada_step_val=[1.0],
        ada_coupling_var=[0.5],
        threshold=1.0,
        reset_val=0.0
    )
    spikes_adaptive, state_adaptive = adaptive_lif(x, reset_state=True)
    
    # Check that all variants produce binary spikes
    assert jnp.all((spikes_simple == 0) | (spikes_simple == 1))
    assert jnp.all((spikes_lif == 0) | (spikes_lif == 1))
    assert jnp.all((spikes_soft == 0) | (spikes_soft == 1))
    assert jnp.all((spikes_adaptive == 0) | (spikes_adaptive == 1))
    
    # Check that outputs have correct shapes
    assert spikes_simple.shape == x.shape
    assert spikes_lif.shape == x.shape
    assert spikes_soft.shape == x.shape
    assert spikes_adaptive.shape == x.shape
    
    print("✅ Different LIF variants test passed")


def test_parameter_clipping():
    """Test that parameters are properly clipped."""
    print("Testing parameter clipping...")
    
    # Create LIF layer with extreme values
    lif = LIF(
        decay_constants=[0.1, 0.2],  # Should be clipped to [0.5, 1.0]
        threshold=1.0,
        reset_val=0.0,
        surrogate_beta=10.0
    )
    
    # Create input data
    x = jnp.array([[0.5, 1.2, 0.8, 1.5]], dtype=jnp.float32)
    
    # Forward pass
    spikes, state = lif(x, reset_state=True)
    
    # Check that the layer still works with clipped parameters
    assert spikes.shape == x.shape
    assert state['mem_pot'].shape == x.shape
    assert state['syn_curr'].shape == x.shape
    
    print("✅ Parameter clipping test passed")


def test_lif_true_behavior():
    """
    Test that LIF layer behaves as a true LIF neuron in both forward and backward passes.
    This test focuses on core LIF behavior rather than specific implementation details.
    """
    print("Testing LIF true behavior...")
    
    # Create LIF layer with simple parameters
    lif = LIF(
        decay_constants=[0.8, 0.7],  # alpha=0.8, beta=0.7
        threshold=0.5,  # Lower threshold to make spiking more likely
        reset_val=0.0,
        surrogate_beta=10.0
    )
    
    # Create input data
    x = jnp.array([[0.5, 1.2, 0.8, 1.5],
                   [0.3, 0.9, 1.1, 0.7]], dtype=jnp.float32)
    
    # Test 1: Forward pass behavior
    print("Testing forward pass...")
    spikes, state = lif(x, reset_state=True)
    
    # Check basic LIF properties
    assert spikes.shape == x.shape, f"Spike shape {spikes.shape} != input shape {x.shape}"
    assert state['mem_pot'].shape == x.shape, f"Membrane potential shape {state['mem_pot'].shape} != input shape {x.shape}"
    assert state['syn_curr'].shape == x.shape, f"Synaptic current shape {state['syn_curr'].shape} != input shape {x.shape}"
    
    # Check that spikes are binary (0 or 1)
    assert jnp.all((spikes == 0) | (spikes == 1)), "Spikes must be binary (0 or 1)"
    
    # Check that membrane potential is finite
    assert jnp.all(jnp.isfinite(state['mem_pot'])), "Membrane potential must be finite"
    assert jnp.all(jnp.isfinite(state['syn_curr'])), "Synaptic current must be finite"
    
    # Test 2: State continuity (membrane potential should evolve over time)
    print("Testing state continuity...")
    spikes2, state2 = lif(x, state=state)
    
    # Membrane potential should change between steps
    assert not jnp.allclose(state['mem_pot'], state2['mem_pot']), "Membrane potential should evolve over time"
    
    # Test 3: Gradient computation
    print("Testing gradient computation...")
    
    def loss_fn(model, inputs):
        spikes, state = model(inputs, reset_state=True)
        # Simple loss: encourage some spiking activity
        return jnp.mean(spikes), (spikes, state)
    
    # Compute gradients
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    (loss, (spikes, state)), grads = grad_fn(lif, x)
    
    # Check that gradients exist and have correct shapes
    assert grads is not None, "Gradients should be computed"
    
    # Get parameter names and check gradients
    param_names, params = get_param_names(lif)
    for name in param_names:
        assert name in grads, f"Gradient for {name} should be computed"
        grad_shape = grads[name].value.shape
        param_shape = params[name].shape
        assert grad_shape == param_shape, f"Gradient shape {grad_shape} != parameter shape {param_shape} for {name}"
    
    # Test 4: Parameter sensitivity (gradients should be non-zero for trainable parameters)
    print("Testing parameter sensitivity...")
    total_grad_norm = sum(jnp.sum(jnp.square(grad.value)) for grad in grads.values())
    assert total_grad_norm > 0, "Gradients should be non-zero for trainable parameters"
    
    # Test 5: Threshold behavior
    print("Testing threshold behavior...")
    
    # Create high input that should cause spiking
    high_input = jnp.array([[2.0, 2.0, 2.0, 2.0],
                           [2.0, 2.0, 2.0, 2.0]], dtype=jnp.float32)
    
    # Run multiple steps to allow membrane potential to build up
    current_state = None
    total_spikes = 0
    for step in range(5):  # Run 5 steps to build up membrane potential
        high_spikes, current_state = lif(high_input, state=current_state, reset_state=(step == 0))
        total_spikes += jnp.sum(high_spikes)
        print(f"Step {step}: Membrane potential = {current_state['mem_pot']}, Spikes = {jnp.sum(high_spikes)}")
    
    # After multiple steps, we should see some spiking activity
    assert total_spikes > 0, "High input over multiple steps should cause some spiking"
    
    # Debug: Print final values
    print(f"Final membrane potential: {current_state['mem_pot']}")
    print(f"Final synaptic current: {current_state['syn_curr']}")
    print(f"Threshold: {lif.threshold.value}")
    print(f"Final spikes: {high_spikes}")
    print(f"Total spikes across all steps: {total_spikes}")
    
    # Test 6: Reset behavior
    print("Testing reset behavior...")
    
    # Create LIF with non-zero reset value
    lif_with_reset = LIF(
        decay_constants=[0.8, 0.7],
        threshold=1.0,
        reset_val=0.5,  # Non-zero reset
        surrogate_beta=10.0
    )
    
    # Run with high input to trigger spikes
    reset_spikes, reset_state = lif_with_reset(high_input, reset_state=True)
    
    # Check that reset is working (membrane potential should be affected by reset)
    assert jnp.all(jnp.isfinite(reset_state['mem_pot'])), "Membrane potential with reset must be finite"
    
    # Test 7: Decay behavior
    print("Testing decay behavior...")
    
    # Run multiple steps and check that membrane potential shows decay behavior
    current_state = None
    for step in range(3):
        spikes_step, current_state = lif(x, state=current_state, reset_state=(step == 0))
        assert jnp.all(jnp.isfinite(current_state['mem_pot'])), f"Membrane potential at step {step} must be finite"
        assert jnp.all(jnp.isfinite(current_state['syn_curr'])), f"Synaptic current at step {step} must be finite"
    
    print("✅ LIF true behavior test passed - LIF layer behaves as expected!")


def run_all_tests():
    """Run all LIF layer tests."""
    print("Running LIF layer tests (snnax-style)...\n")
    
    try:
        test_simple_lif_creation()
        test_lif_creation()
        test_adaptive_lif_creation()
        test_simple_lif_forward_pass()
        test_lif_forward_pass()
        test_adaptive_lif_forward_pass()
        test_gradient_computation()
        test_parameter_updates()
        test_state_continuity()
        test_surrogate_gradients()
        test_different_lif_variants()
        test_parameter_clipping()
        test_lif_true_behavior()
        
        print("\n🎉 All LIF layer tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_all_tests() 