"""
Training package for the Hyper framework.

Contains training functions, evaluation, and model checkpointing.
"""

# Import training functions with error handling
try:
    from .training_functions import train_step, evaluate_model
except ImportError as e:
    # If relative import fails, try absolute import
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from training.training_functions import train_step, evaluate_model

# Import checkpointing functions
from .checkpointing import save_checkpoint, restore_checkpoint, save_model_checkpoint

__all__ = [
    'train_step',
    'evaluate_model',
    'save_checkpoint',
    'restore_checkpoint',
    'save_model_checkpoint',

] 