"""
Constants for the Hyper framework models.

Defines action types, class counts, and other model-related constants.
"""

# Action constants for classification
ACTION_HOLD = 0
ACTION_BUY_CALL = 1
ACTION_BUY_PUT = 2
NUM_CLASSES = 3

# Model configuration constants
DEFAULT_D_MODEL = 256
DEFAULT_NUM_HEADS = 8
DEFAULT_D_FF = 1024
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_NUM_LAYERS = 4
DEFAULT_MAX_LEN = 5000 