"""
Models package for the Hyper framework.

Contains the GPT classifier model, LIF layers, and related neural network components.
"""

from .gpt_classifier import GPTClassifier, TransformerBlock, PositionalEncoding
from .lif_layer import (
    SimpleLIF, LIF, LIFSoftReset, AdaptiveLIF,
    superspike_surrogate, sigmoid_surrogate, piecewise_surrogate
)
from .constants import ACTION_HOLD, ACTION_BUY_CALL, ACTION_BUY_PUT, NUM_CLASSES

__all__ = [
    'GPTClassifier',
    'TransformerBlock', 
    'PositionalEncoding',
    'SimpleLIF',
    'LIF',
    'LIFSoftReset',
    'AdaptiveLIF',
    'superspike_surrogate',
    'sigmoid_surrogate',
    'piecewise_surrogate',
    'ACTION_HOLD',
    'ACTION_BUY_CALL',
    'ACTION_BUY_PUT',
    'NUM_CLASSES'
] 