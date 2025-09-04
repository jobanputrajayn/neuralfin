"""
Checkpointing utilities for the Hyper framework.

Contains functions for saving and restoring model checkpoints using Orbax with Flax NNX.
"""

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from flax import nnx


def get_checkpoint_manager(checkpoint_dir, max_to_keep=1):
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep, 
        create=True
    )
    return ocp.CheckpointManager(
        checkpoint_dir,
        options=options,
    )


def save_checkpoint(
    checkpoint_dir: Path,
    model: Any,  # NNX model instance
    opt_state: Any,
    scaler_mean: float,
    scaler_std: float,
    step: int,
    learning_rate: float,
    max_to_keep: int = 1
) -> str:
    """
    Save model state, optimizer state, learning rate, and scalers using Orbax CheckpointManager.
    Args:
        checkpoint_dir: Directory to save checkpoints
        model: NNX model instance (will be split to get params and state)
        opt_state: Optimizer state (can be None)
        scaler_mean: Mean value for data scaling
        scaler_std: Standard deviation for data scaling
        step: Current training step
        learning_rate: Current learning rate
        max_to_keep: Maximum number of checkpoints to keep
    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    params, state = nnx.split(model)
    
    # Create Composite args for saving
    composite_args = {
        'params': ocp.args.PyTreeSave(params),
        'state': ocp.args.PyTreeSave(state),
        'scaler_mean': ocp.args.ArraySave(float(scaler_mean)),
        'scaler_std': ocp.args.ArraySave(float(scaler_std)),
        'step': ocp.args.ArraySave(step),
        'learning_rate': ocp.args.ArraySave(float(learning_rate)),
    }
    
    # Only add opt_state if it's not None
    if opt_state is not None:
        composite_args['opt_state'] = ocp.args.PyTreeSave(opt_state)
    
    manager = get_checkpoint_manager(checkpoint_dir, max_to_keep)
    
    # Save using Composite args
    manager.save(
        step,
        args=ocp.args.Composite(**composite_args)
    )
    manager.wait_until_finished()  # Ensure save is completed
    print(f"Checkpoint saved at step {step} to {checkpoint_dir}")
    return str(checkpoint_dir)


def restore_checkpoint(
    checkpoint_dir: Path,
    model,
    step: Optional[int] = None,
    learning_rate: Optional[float] = 1e-3,
    create_optimizer_fn=None
) -> Tuple[Optional[Any], Optional[Any], Optional[float], Optional[float], Optional[int], Optional[float], Optional[Any]]:
    """
    Restore model state, optimizer state, learning rate, and scalers from checkpoint.
    Args:
        checkpoint_dir: Directory containing checkpoint
        model: Model instance (will be split to get abstract structure)
        step: Specific step to restore from (if None, uses latest step)
        learning_rate: Learning rate to use for dummy optimizer (if needed)
        create_optimizer_fn: Function(model, learning_rate) -> optimizer, for abstract opt_state
    Returns:
        Tuple of (params, state, scaler_mean, scaler_std, step, learning_rate, opt_state)
        Returns None values if checkpoint not found
    """
    checkpoint_dir = Path(checkpoint_dir).resolve()
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return None, None, None, None, None, None, None
    
    manager = get_checkpoint_manager(checkpoint_dir)
    if step is None:
        step = manager.latest_step()
        if step is None:
            print(f"No checkpoints found in {checkpoint_dir}")
            return None, None, None, None, None, None, None
    
    # Use the provided model instance to get abstract structure
    print(f"Using provided model instance: {type(model).__name__}")
    abstract_params, abstract_state = nnx.split(model)
    
    # Create dummy optimizer for abstract opt_state if needed
    opt_state_abstract = None
    if create_optimizer_fn is not None:
        try:
            dummy_optimizer = create_optimizer_fn(model, learning_rate)
            opt_state_abstract = dummy_optimizer.state
        except Exception as e:
            print(f"Warning: Could not create dummy optimizer: {e}")
    
    # Create composite args for restoration
    composite_args = {
        'params': ocp.args.PyTreeRestore(abstract_params),
        'state': ocp.args.PyTreeRestore(abstract_state),
        'scaler_mean': ocp.args.ArrayRestore(jnp.array(0.0, dtype=jnp.float32)),
        'scaler_std': ocp.args.ArrayRestore(jnp.array(1.0, dtype=jnp.float32)),
        'step': ocp.args.ArrayRestore(jnp.array(0, dtype=jnp.int32)),
        'learning_rate': ocp.args.ArrayRestore(jnp.array(0.0, dtype=jnp.float32)),
    }
    
    if opt_state_abstract is not None:
        composite_args['opt_state'] = ocp.args.PyTreeRestore(opt_state_abstract)
    
    try:
        checkpoint_data = manager.restore(
            step,
            args=ocp.args.Composite(**composite_args)
        )
        manager.wait_until_finished()  # Ensure restore is completed
    except Exception as e:
        print(f"Error restoring checkpoint: {e}")
        print(f"Checkpoint directory: {checkpoint_dir}")
        print(f"Step: {step}")
        raise
    
    # Extract restored data
    params = checkpoint_data['params']
    state = checkpoint_data['state']
    scaler_mean = float(checkpoint_data['scaler_mean'])
    scaler_std = float(checkpoint_data['scaler_std'])
    restored_step = int(checkpoint_data['step'])
    restored_lr = float(checkpoint_data['learning_rate'])
    opt_state = checkpoint_data.get('opt_state', None)
    
    print(f"Checkpoint restored from {checkpoint_dir} at step {restored_step}")
    return params, state, scaler_mean, scaler_std, restored_step, restored_lr, opt_state


def save_model_checkpoint(ckpt_mgr, step, model, scaler_mean, scaler_std, learning_rate):
    if hasattr(ckpt_mgr, 'directory'):
        checkpoint_dir = Path(ckpt_mgr.directory)
    else:
        checkpoint_dir = Path("./checkpoints")
    save_checkpoint(checkpoint_dir, model, None, scaler_mean, scaler_std, step, learning_rate)


 