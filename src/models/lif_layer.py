"""
LIF (Leaky Integrate-and-Fire) Layer for Spiking Neural Networks

This module implements LIF layers using NNX, based on the actual snnax implementation.
The LIF layer implements the standard leaky integrate-and-fire neuron dynamics.

Third-Party Attribution:
- Original implementation: snnax library (https://github.com/neuromorphs/snnax)
- License: MIT License (compatible)
- Adapted for Flax NNX framework
- See docs/ATTRIBUTIONS.md for full attribution details
"""

import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax import custom_jvp
from flax import nnx
from flax.nnx import Param, State
from typing import Optional, Tuple, Union, Callable, Sequence
import numpy as np


# Custom Variable types for LIF state
class MembranePotential(nnx.Variable): pass
class SynapticCurrent(nnx.Variable): pass
class SpikeOutput(nnx.Variable): pass
class AdaptiveVariable(nnx.Variable): pass


# Surrogate gradient functions (from snnax)
def superspike_surrogate(beta: float = 10.):
    """
    Implementation of the superspike surrogate gradient function.
    
    Args:
        beta: Parameter to control the steepness of the surrogate gradient
        
    Returns:
        A function that returns the surrogate gradient of the heaviside function
    """
    @custom_jvp
    def heaviside_with_superspike_surrogate(x):
        return jnp.heaviside(x, 1.)

    @heaviside_with_superspike_surrogate.defjvp
    def f_jvp(primals, tangents):
        x, = primals
        x_dot, = tangents
        primal_out = heaviside_with_superspike_surrogate(x)
        tangent_out = x_dot / (1. + beta * jnp.abs(x))
        return primal_out, tangent_out
    
    return heaviside_with_superspike_surrogate


def sigmoid_surrogate(beta: float = 1.):
    """
    Implementation of the sigmoidal surrogate gradient function.
    
    Args:
        beta: Parameter to control the steepness of the surrogate gradient
        
    Returns:
        A function that returns the surrogate gradient of the heaviside function
    """
    @custom_jvp
    def heaviside_with_sigmoid_surrogate(x):
        return jnp.heaviside(x, 1.)

    @heaviside_with_sigmoid_surrogate.defjvp
    def f_jvp(primals, tangents):
        x, = primals
        x_dot, = tangents
        primal_out = heaviside_with_sigmoid_surrogate(x)
        tangent_out = jnn.sigmoid(x * beta) 
        tangent_out *= (1. - beta * jnn.sigmoid(x * beta)) * x_dot 
        return primal_out, tangent_out

    return heaviside_with_sigmoid_surrogate


def piecewise_surrogate(beta: float = 0.5):
    """
    Implementation of the piecewise surrogate gradient function.
    
    Args:
        beta: Parameter to control the steepness of the surrogate gradient
        
    Returns:
        A function that returns the surrogate gradient of the heaviside function
    """
    @custom_jvp
    def heaviside_with_piecewise_surrogate(x):
        return jnp.heaviside(x, 1.)

    @heaviside_with_piecewise_surrogate.defjvp
    def f_jvp(primals, tangents):
        x, = primals
        x_dot, = tangents
        primal_out = heaviside_with_piecewise_surrogate(x)
        tangent_out = jnp.where((x > -beta) & (x < beta), x_dot, 0.0)
        return primal_out, tangent_out

    return heaviside_with_piecewise_surrogate


class SimpleLIF(nnx.Module):
    """
    Simple implementation of a layer of leaky integrate-and-fire neurons 
    which does not make explicit use of synaptic currents.
    Requires one decay constant to simulate membrane potential leak.
    
    Based on snnax implementation but adapted for NNX.
    """
    
    def __init__(
        self,
        decay_constants: Union[Sequence[float], jnp.ndarray],
        spike_fn: Optional[Callable] = None,
        threshold: float = 1.0,
        stop_reset_grad: bool = True,
        reset_val: Optional[float] = None,
        surrogate_beta: float = 10.0,
        rngs: Optional[nnx.Rngs] = None,
    ):
        super().__init__()
        
        # Initialize trainable parameters (gradable in original snnax)
        if isinstance(decay_constants, (list, tuple)):
            self.decay_constants = nnx.Param(jnp.array(decay_constants, dtype=jnp.float32))
        else:
            self.decay_constants = nnx.Param(decay_constants)
            
        # Make threshold trainable (optional in snnax)
        self.threshold = nnx.Param(jnp.array(threshold, dtype=jnp.float32))
        
        # Make reset_val trainable (optional in snnax)
        if reset_val is not None:
            self.reset_val = nnx.Param(jnp.array(reset_val, dtype=jnp.float32))
        else:
            self.reset_val = None
            
        # Make surrogate beta trainable (gradable in snnax)
        self.surrogate_beta = nnx.Param(jnp.array(surrogate_beta, dtype=jnp.float32))
        
        self.stop_reset_grad = stop_reset_grad
        
        # Set default spike function if none provided - create with initial beta value
        if spike_fn is None:
            self.spike_fn = superspike_surrogate(surrogate_beta)
        else:
            self.spike_fn = spike_fn
            
        # Initialize state variables using nnx.Variables
        self.mem_pot = MembranePotential(jnp.zeros((1, 1), dtype=jnp.float32))  # Will be resized dynamically
        self.spike_output = SpikeOutput(jnp.zeros((1, 1), dtype=jnp.float32))    # Will be resized dynamically
    
    def _reset_state(self, batch_size: int, hidden_size: int):
        """Reset internal state variables."""
        self.mem_pot.value = jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)
        self.spike_output.value = jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)
    
    def __call__(self, x: jnp.ndarray, state: Optional[dict] = None, reset_state: bool = False) -> Tuple[jnp.ndarray, dict]:
        """
        Forward pass through the SimpleLIF layer.
        Follows snnax sequence: update membrane potential, generate spikes, handle reset.
        
        Args:
            x: Input tensor of shape (batch_size, hidden_size)
            state: Optional state dictionary with keys ['mem_pot', 'spike_output']
            reset_state: Whether to reset internal state (ignored if state is provided)
            
        Returns:
            Tuple of (spikes, state_dict) where:
            - spikes: Binary spike tensor of shape (batch_size, hidden_size)
            - state_dict: Dictionary containing internal state variables
        """
        batch_size, hidden_size = x.shape
        
        # Initialize or get state
        if state is None:
            if reset_state or self.mem_pot.value.shape != (batch_size, hidden_size):
                self._reset_state(batch_size, hidden_size)
            mem_pot = self.mem_pot.value
            spike_output = self.spike_output.value
        else:
            mem_pot = state['mem_pot']
            spike_output = state['spike_output']
        
        # Get decay constant (alpha) - trainable parameter - use clamp like snnax
        alpha = jax.lax.clamp(0.5, self.decay_constants[0], 1.0)
        
        # Update membrane potential (snnax sequence)
        mem_pot = alpha * mem_pot + (1.0 - alpha) * x
        
        # Generate spikes using surrogate function (created during initialization)
        spike_output = self.spike_fn(mem_pot - self.threshold)
        
        # Handle reset (snnax sequence)
        if self.reset_val is None:
            reset_pot = mem_pot * spike_output
        else:
            reset_val = jnn.softplus(self.reset_val.value)
            reset_pot = reset_val * spike_output
        
        # Optionally stop gradient propagation through refractory potential
        if self.stop_reset_grad:
            refractory_potential = jax.lax.stop_gradient(reset_pot)
        else:
            refractory_potential = reset_pot
        
        # Update membrane potential (subtract refractory potential)
        mem_pot = mem_pot - refractory_potential
        
        # Update internal state if not using external state
        if state is None:
            self.mem_pot.value = mem_pot
            self.spike_output.value = spike_output
        
        # Prepare state dictionary
        state_dict = {
            'mem_pot': mem_pot,
            'spike_output': spike_output
        }
        
        return spike_output, state_dict
    
    def reset(self):
        """Reset all internal state variables."""
        self.mem_pot.value = jnp.zeros_like(self.mem_pot.value)
        self.spike_output.value = jnp.zeros_like(self.spike_output.value)


class LIF(nnx.Module):
    """
    Implementation of a leaky integrate-and-fire neuron with synaptic currents.
    Requires two decay constants to describe decay of membrane potential and synaptic current.
    
    Based on snnax implementation but adapted for NNX.
    """
    
    def __init__(
        self,
        decay_constants: Union[Sequence[float], jnp.ndarray],
        spike_fn: Optional[Callable] = None,
        threshold: float = 1.0,
        stop_reset_grad: bool = True,
        reset_val: Optional[float] = None,
        surrogate_beta: float = 10.0,
        rngs: Optional[nnx.Rngs] = None,
    ):
        super().__init__()
        
        # Initialize trainable parameters (gradable in original snnax)
        if isinstance(decay_constants, (list, tuple)):
            self.decay_constants = nnx.Param(jnp.array(decay_constants, dtype=jnp.float32))
        else:
            self.decay_constants = nnx.Param(decay_constants)
            
        # Make threshold trainable (optional in snnax)
        self.threshold = nnx.Param(jnp.array(threshold, dtype=jnp.float32))
        
        # Make reset_val trainable (optional in snnax)
        if reset_val is not None:
            self.reset_val = nnx.Param(jnp.array(reset_val, dtype=jnp.float32))
        else:
            self.reset_val = None
            
        # Make surrogate beta trainable (gradable in snnax)
        self.surrogate_beta = nnx.Param(jnp.array(surrogate_beta, dtype=jnp.float32))
        
        self.stop_reset_grad = stop_reset_grad
        
        # Set default spike function if none provided - create with initial beta value
        if spike_fn is None:
            self.spike_fn = superspike_surrogate(surrogate_beta)
        else:
            self.spike_fn = spike_fn
            
        # Initialize state variables using nnx.Variables
        self.mem_pot = MembranePotential(jnp.zeros((1, 1), dtype=jnp.float32))  # Will be resized dynamically
        self.syn_curr = SynapticCurrent(jnp.zeros((1, 1), dtype=jnp.float32))   # Will be resized dynamically
        self.spike_output = SpikeOutput(jnp.zeros((1, 1), dtype=jnp.float32))   # Will be resized dynamically
    
    def _reset_state(self, batch_size: int, hidden_size: int):
        """Reset internal state variables."""
        self.mem_pot.value = jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)
        self.syn_curr.value = jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)
        self.spike_output.value = jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)
    
    def __call__(self, x: jnp.ndarray, reset_state: bool = False) -> jnp.ndarray:
        """
        Forward pass through the LIF layer.
        Supports input x of shape (..., hidden) or (..., time, hidden),
        where ... can be any number of leading axes (e.g., device, batch, etc).
        For (..., time, hidden), processes the sequence along the temporal axis, maintaining state across time.
        For (..., hidden), keeps the original behavior.
        State is managed internally via nnx.Variables (no state dicts).
        If reset_state is False, state is carried over between calls for each batch sample.
        """
        if x.ndim < 2:
            raise ValueError(f"LIF layer expects input of shape (..., hidden) or (..., time, hidden), got {x.shape}")
        # (..., hidden)
        if x.ndim == 2:
            return self._call_2d(x, reset_state)
        # (..., time, hidden)
        elif x.ndim >= 3:
            leading_shape = x.shape[:-2]
            time = x.shape[-2]
            hidden_size = x.shape[-1]
            x_flat = x.reshape((-1, time, hidden_size))
            # Prepare initial state for each sample in the flattened batch
            batch_size = x_flat.shape[0]
            if reset_state or self.mem_pot.value.shape != (batch_size, hidden_size):
                mem_pot_init = jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)
                syn_curr_init = jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)
                spike_output_init = jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)
            else:
                mem_pot_init = self.mem_pot.value
                syn_curr_init = self.syn_curr.value
                spike_output_init = self.spike_output.value
            def scan_lif(x_seq, mem_pot, syn_curr, spike_output):
                def step(carry, x_t):
                    mem_pot, syn_curr, spike_output = carry
                    alpha = jax.lax.clamp(0.5, self.decay_constants[0], 1.0)
                    beta = jax.lax.clamp(0.5, self.decay_constants[1], 1.0)
                    if self.reset_val is None:
                        reset_pot = mem_pot * spike_output
                    else:
                        reset_pot = (mem_pot - self.reset_val.value) * spike_output
                    refractory_potential = jax.lax.stop_gradient(reset_pot) if self.stop_reset_grad else reset_pot
                    mem_pot = mem_pot - refractory_potential
                    mem_pot = alpha * mem_pot + (1.0 - alpha) * syn_curr
                    syn_curr = beta * syn_curr + (1.0 - beta) * x_t
                    spike_output = self.spike_fn(mem_pot - self.threshold)
                    return (mem_pot, syn_curr, spike_output), spike_output
                init_carry = (mem_pot, syn_curr, spike_output)
                (final_mem_pot, final_syn_curr, final_spike_output), spikes = jax.lax.scan(step, init_carry, x_seq)
                return spikes, (final_mem_pot, final_syn_curr, final_spike_output)
            # vmap over flattened batch
            spikes, final_states = jax.vmap(scan_lif, in_axes=(0, 0, 0, 0))(x_flat, mem_pot_init, syn_curr_init, spike_output_init)
            # Update internal state with final state from each batch sample
            self.mem_pot.value = final_states[0]
            self.syn_curr.value = final_states[1]
            self.spike_output.value = final_states[2]
            # Reshape output back to original leading shape
            return spikes.reshape(leading_shape + (time, hidden_size))
        else:
            raise ValueError(f"LIF layer expects input of shape (..., hidden) or (..., time, hidden), got {x.shape}")

    def _call_2d(self, x: jnp.ndarray, reset_state: bool = False) -> jnp.ndarray:
        """
        Forward pass through the LIF layer for 2D input (batch_size, hidden_size).
        State is managed internally via nnx.Variables (no state dicts).
        """
        batch_size, hidden_size = x.shape
        if reset_state or self.mem_pot.value.shape != (batch_size, hidden_size):
            self._reset_state(batch_size, hidden_size)
        mem_pot = self.mem_pot.value
        syn_curr = self.syn_curr.value
        spike_output = self.spike_output.value
        # Get decay constants (alpha, beta)
        alpha = jax.lax.clamp(0.5, self.decay_constants[0], 1.0)
        beta = jax.lax.clamp(0.5, self.decay_constants[1], 1.0)
        # Handle reset
        if self.reset_val is None:
            reset_pot = mem_pot * spike_output
        else:
            reset_pot = (mem_pot - self.reset_val.value) * spike_output
        refractory_potential = jax.lax.stop_gradient(reset_pot) if self.stop_reset_grad else reset_pot
        mem_pot = mem_pot - refractory_potential
        mem_pot = alpha * mem_pot + (1.0 - alpha) * syn_curr
        syn_curr = beta * syn_curr + (1.0 - beta) * x
        spike_output = self.spike_fn(mem_pot - self.threshold)
        # Update internal state
        self.mem_pot.value = mem_pot
        self.syn_curr.value = syn_curr
        self.spike_output.value = spike_output
        return spike_output
    
    def reset(self):
        """Reset all internal state variables."""
        self.mem_pot.value = jnp.zeros_like(self.mem_pot.value)
        self.syn_curr.value = jnp.zeros_like(self.syn_curr.value)
        self.spike_output.value = jnp.zeros_like(self.spike_output.value)


class LIFSoftReset(LIF):
    """
    Similar to LIF but reset is additive (relative) rather than absolute.
    If the neurons spikes: V -> V_reset where V_reset is the parameter reset_val.
    
    Based on snnax implementation but adapted for NNX.
    """
    
    def __call__(self, x: jnp.ndarray, state: Optional[dict] = None, reset_state: bool = False) -> Tuple[jnp.ndarray, dict]:
        """
        Forward pass through the LIFSoftReset layer.
        Follows snnax sequence: handle reset first, then update membrane potential and synaptic current, then generate spikes.
        
        Args:
            x: Input tensor of shape (batch_size, hidden_size)
            state: Optional state dictionary with keys ['mem_pot', 'syn_curr', 'spike_output']
            reset_state: Whether to reset internal state (ignored if state is provided)
            
        Returns:
            Tuple of (spikes, state_dict) where:
            - spikes: Binary spike tensor of shape (batch_size, hidden_size)
            - state_dict: Dictionary containing internal state variables
        """
        batch_size, hidden_size = x.shape
        
        # Initialize or get state
        if state is None:
            if reset_state or self.mem_pot.value.shape != (batch_size, hidden_size):
                self._reset_state(batch_size, hidden_size)
            mem_pot = self.mem_pot.value
            syn_curr = self.syn_curr.value
            spike_output = self.spike_output.value
        else:
            mem_pot = state['mem_pot']
            syn_curr = state['syn_curr']
            spike_output = state['spike_output']
        
        # Handle reset first (additive/relative reset) - snnax sequence
        if self.reset_val is None:
            reset_pot = spike_output
        else:
            reset_pot = self.reset_val.value * spike_output
        
        # Optionally stop gradient propagation through refractory potential
        if self.stop_reset_grad:
            refractory_potential = jax.lax.stop_gradient(reset_pot)
        else:
            refractory_potential = reset_pot
        
        # Update membrane potential (subtract refractory potential)
        mem_pot = mem_pot - refractory_potential
        
        # Get decay constants (alpha, beta) - trainable parameters - use clamp like snnax
        alpha = jax.lax.clamp(0.5, self.decay_constants[0], 1.0)
        beta = jax.lax.clamp(0.5, self.decay_constants[1], 1.0)
        
        # Update membrane potential and synaptic current (snnax sequence)
        mem_pot = alpha * mem_pot + (1.0 - alpha) * syn_curr
        syn_curr = beta * syn_curr + (1.0 - beta) * x
        
        # Generate spikes using surrogate function (created during initialization)
        spike_output = self.spike_fn(mem_pot - self.threshold)
        
        # Update internal state if not using external state
        if state is None:
            self.mem_pot.value = mem_pot
            self.syn_curr.value = syn_curr
            self.spike_output.value = spike_output
        
        # Prepare state dictionary
        state_dict = {
            'mem_pot': mem_pot,
            'syn_curr': syn_curr,
            'spike_output': spike_output
        }
        
        return spike_output, state_dict


class AdaptiveLIF(nnx.Module):
    """
    Implementation of an adaptive exponential leaky integrate-and-fire neuron.
    
    Based on snnax implementation but adapted for NNX.
    """
    
    def __init__(
        self,
        decay_constants: Union[Sequence[float], jnp.ndarray],
        ada_decay_constant: Union[Sequence[float], jnp.ndarray] = [0.8],
        ada_step_val: Union[Sequence[float], jnp.ndarray] = [1.0],
        ada_coupling_var: Union[Sequence[float], jnp.ndarray] = [0.5],
        spike_fn: Optional[Callable] = None,
        threshold: float = 1.0,
        stop_reset_grad: bool = True,
        reset_val: Optional[float] = None,
        surrogate_beta: float = 10.0,
        rngs: Optional[nnx.Rngs] = None,
    ):
        super().__init__()
        
        # Initialize trainable parameters (gradable in original snnax)
        if isinstance(decay_constants, (list, tuple)):
            self.decay_constants = nnx.Param(jnp.array(decay_constants, dtype=jnp.float32))
        else:
            self.decay_constants = nnx.Param(decay_constants)
            
        if isinstance(ada_decay_constant, (list, tuple)):
            self.ada_decay_constant = nnx.Param(jnp.array(ada_decay_constant, dtype=jnp.float32))
        else:
            self.ada_decay_constant = nnx.Param(ada_decay_constant)
            
        if isinstance(ada_step_val, (list, tuple)):
            self.ada_step_val = nnx.Param(jnp.array(ada_step_val, dtype=jnp.float32))
        else:
            self.ada_step_val = nnx.Param(ada_step_val)
            
        if isinstance(ada_coupling_var, (list, tuple)):
            self.ada_coupling_var = nnx.Param(jnp.array(ada_coupling_var, dtype=jnp.float32))
        else:
            self.ada_coupling_var = nnx.Param(ada_coupling_var)
            
        # Make threshold trainable (optional in snnax)
        self.threshold = nnx.Param(jnp.array(threshold, dtype=jnp.float32))
        
        # Make reset_val trainable (optional in snnax)
        if reset_val is not None:
            self.reset_val = nnx.Param(jnp.array(reset_val, dtype=jnp.float32))
        else:
            self.reset_val = None
            
        # Make surrogate beta trainable (gradable in snnax)
        self.surrogate_beta = nnx.Param(jnp.array(surrogate_beta, dtype=jnp.float32))
        
        self.stop_reset_grad = stop_reset_grad
        
        # Set default spike function if none provided - create with initial beta value
        if spike_fn is None:
            self.spike_fn = superspike_surrogate(surrogate_beta)
        else:
            self.spike_fn = spike_fn
            
        # Initialize state variables using nnx.Variables
        self.mem_pot = MembranePotential(jnp.zeros((1, 1), dtype=jnp.float32))  # Will be resized dynamically
        self.ada_var = AdaptiveVariable(jnp.zeros((1, 1), dtype=jnp.float32))    # Will be resized dynamically
        self.spike_output = SpikeOutput(jnp.zeros((1, 1), dtype=jnp.float32))    # Will be resized dynamically
    
    def _reset_state(self, batch_size: int, hidden_size: int):
        """Reset internal state variables."""
        self.mem_pot.value = jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)
        self.ada_var.value = jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)
        self.spike_output.value = jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)
    
    def __call__(self, x: jnp.ndarray, state: Optional[dict] = None, reset_state: bool = False) -> Tuple[jnp.ndarray, dict]:
        """
        Forward pass through the AdaptiveLIF layer.
        Follows snnax sequence: calculate membrane potential, generate spikes, calculate adaptive dynamics, handle reset.
        
        Args:
            x: Input tensor of shape (batch_size, hidden_size)
            state: Optional state dictionary with keys ['mem_pot', 'ada_var', 'spike_output']
            reset_state: Whether to reset internal state (ignored if state is provided)
            
        Returns:
            Tuple of (spikes, state_dict) where:
            - spikes: Binary spike tensor of shape (batch_size, hidden_size)
            - state_dict: Dictionary containing internal state variables
        """
        batch_size, hidden_size = x.shape
        
        # Initialize or get state
        if state is None:
            if reset_state or self.mem_pot.value.shape != (batch_size, hidden_size):
                self._reset_state(batch_size, hidden_size)
            mem_pot = self.mem_pot.value
            ada_var = self.ada_var.value
            spike_output = self.spike_output.value
        else:
            mem_pot = state['mem_pot']
            ada_var = state['ada_var']
            spike_output = state['spike_output']
        
        # Get trainable parameters - use clamp like snnax
        alpha = jax.lax.clamp(0.5, self.decay_constants[0], 1.0)
        beta = jax.lax.clamp(0.5, self.ada_decay_constant[0], 1.0)
        a = jax.lax.clamp(-1.0, self.ada_coupling_var[0], 1.0)
        b = jax.lax.clamp(0.0, self.ada_step_val[0], 2.0)
        
        # Calculate membrane potential (snnax sequence)
        mem_pot = alpha * mem_pot + (1.0 - alpha) * (x + ada_var)
        
        # Generate spikes using surrogate function (created during initialization)
        spike_output = self.spike_fn(mem_pot - self.threshold)
        
        # Calculate adaptive part of dynamics (snnax sequence)
        ada_var_new = (1.0 - beta) * a * mem_pot + beta * ada_var - b * jax.lax.stop_gradient(spike_output)
        
        # Handle reset
        if self.reset_val is None:
            reset_pot = mem_pot * spike_output
        else:
            reset_pot = self.reset_val.value * spike_output
        
        # Optionally stop gradient propagation through refractory potential
        if self.stop_reset_grad:
            refractory_potential = jax.lax.stop_gradient(reset_pot)
        else:
            refractory_potential = reset_pot
        
        # Update membrane potential (subtract refractory potential)
        mem_pot = mem_pot - refractory_potential
        
        # Update internal state if not using external state
        if state is None:
            self.mem_pot.value = mem_pot
            self.ada_var.value = ada_var_new
            self.spike_output.value = spike_output
        
        # Prepare state dictionary
        state_dict = {
            'mem_pot': mem_pot,
            'ada_var': ada_var_new,
            'spike_output': spike_output
        }
        
        return spike_output, state_dict
    
    def reset(self):
        """Reset all internal state variables."""
        self.mem_pot.value = jnp.zeros_like(self.mem_pot.value)
        self.ada_var.value = jnp.zeros_like(self.ada_var.value)
        self.spike_output.value = jnp.zeros_like(self.spike_output.value) 