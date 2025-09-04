"""
GPT Classifier Model for Stock Prediction (NNX version)

This module contains the GPT-like transformer model used for stock prediction,
including positional encoding, transformer blocks, and the main classifier.
"""

import jax
import jax.numpy as jnp
import jax.lax
from flax import nnx
from flax.nnx import Param, BatchStat, State, split, merge, update
import numpy as np
from functools import partial
from .constants import NUM_CLASSES
from typing import Optional, Any
from flax.nnx import gelu
import optax
from .lif_layer import LIF


class PositionalEncoding(nnx.Module):
    """Sinusoidal positional encoding for transformer models."""
    def __init__(self, d_model: int, maxlen: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.maxlen = maxlen
        # Precompute positional encodings (static, not a parameter)
        positional_encodings = np.zeros((self.maxlen, self.d_model), dtype=jnp.float32)
        position_indices = np.arange(0, self.maxlen, dtype=np.float32)[:, np.newaxis]
        division_terms = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        positional_encodings[:, 0::2] = np.sin(position_indices * division_terms)
        positional_encodings[:, 1::2] = np.cos(position_indices * division_terms)
        self.pe = jnp.asarray(positional_encodings)

    def __call__(self, x):
        # x: (batch_size, seq_length, embedding_dim)
        positional_encodings = self.pe[:x.shape[1]]  # (seq_length, d_model)
        return x + jax.lax.stop_gradient(positional_encodings)[None, :, :]


class TransformerBlock(nnx.Module):
    """A single Transformer block with multi-head attention and feed-forward network."""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float, rngs: nnx.Rngs):
        super().__init__()
        self.attention = nnx.MultiHeadAttention(
            in_features=d_model,
            num_heads=num_heads,
            qkv_features=d_model,
            out_features=d_model,
            dropout_rate=dropout_rate,
            rngs=rngs
        )
        self.layer_norm1 = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.feedforward_layer1 = nnx.Linear(d_model, d_ff, rngs=rngs)
        
        # Add LIF layer
        self.lif_layer = LIF(
            decay_constants=[0.8, 0.7],
            threshold=1.0,
            reset_val=0.0,
            surrogate_beta=10.0
        )
        
        self.feedforward_layer2 = nnx.Linear(d_ff, d_model, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.layer_norm2 = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.dropout_rate = dropout_rate

    def __call__(self, x):
        # Multi-head attention
        attention_output = self.attention(x, x, x, decode=False)
        x = self.layer_norm1(x + attention_output)
        
        # Feed-forward with both GELU and LIF processing
        feedforward_output = self.feedforward_layer1(x)
        
        # Apply GELU activation first
        feedforward_output = gelu(feedforward_output)
        
        # Apply LIF processing as additional step (now supports 3D input)
        # feedforward_output: (batch_size, seq_length, d_ff)
        lif_spikes = self.lif_layer(feedforward_output, reset_state=True)  # (batch_size, seq_length, d_ff)
        
        # Combine GELU output with LIF spikes
        feedforward_output = feedforward_output + lif_spikes
        
        feedforward_output = self.feedforward_layer2(feedforward_output)
        feedforward_output = self.dropout(feedforward_output)
        x = self.layer_norm2(x + feedforward_output)
        return x


class GPTClassifier(nnx.Module):
    """
    GPT-like classifier for stock prediction (NNX version).
    This model takes sequences of stock prices and predicts trading actions
    (HOLD, BUY_CALL, BUY_PUT) for each stock in the universe.
    """
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = 0.1,
        input_features: int = None,
        num_tickers: int = None,
        maxlen: int = 5000,
        rngs: nnx.Rngs = None,
        learning_rate: float = 1e-4,
        optimizer: Optional[Any] = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.input_features = input_features
        self.num_tickers = num_tickers
        self.maxlen = maxlen
        self.rngs = rngs if rngs is not None else nnx.Rngs(0)
        self.learning_rate = learning_rate

        # Input projection
        self.input_proj = nnx.Linear(input_features, d_model, rngs=self.rngs)
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, maxlen)
        # Input dropout
        self.input_dropout = nnx.Dropout(dropout_rate, rngs=self.rngs)
        # Transformer blocks as a list
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, d_ff, dropout_rate, rngs=self.rngs)
            for _ in range(num_layers)
        ]
        # Output projection: d_model -> num_tickers * num_classes
        self.output_proj = nnx.Linear(d_model, num_tickers * num_classes, rngs=self.rngs)
        
        # Training mode flag
        self._training = False
        
        # Set to evaluation mode by default
        self.eval()

        # Optimizer (nnx.Optimizer encapsulation)
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = nnx.Optimizer(self, optax.adam(self.learning_rate))

    def __call__(self, x, padding_mask: Optional[jnp.ndarray] = None):
        # x: (batch_size, seq_length, num_features) or (n_devices, batch_size, seq_length, num_features)
        input_sequence = x
        # Handle leading device axis for multi-device training
        if input_sequence.ndim == 4:
            # Merge device and batch axes
            input_sequence = input_sequence.reshape((-1,) + input_sequence.shape[2:])
            if padding_mask is not None:
                padding_mask = padding_mask.reshape((-1,) + padding_mask.shape[2:])
        if input_sequence.ndim == 2:
            input_sequence = input_sequence[..., None]
        elif input_sequence.ndim != 3:
            raise ValueError(f"Expected 2D, 3D, or 4D input, got shape {x.shape}")
        # Project input features
        input_sequence = self.input_proj(input_sequence)
        # Add positional encoding
        input_sequence = self.pos_encoding(input_sequence)
        # Dropout
        input_sequence = self.input_dropout(input_sequence)
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            input_sequence = transformer_block(input_sequence)
        # Take last token
        input_sequence = input_sequence[:, -1, :]
        # Apply padding mask if provided
        if padding_mask is not None:
            input_sequence = input_sequence * jnp.expand_dims(padding_mask, axis=-1)
        # Output projection: (batch, d_model) -> (batch, num_tickers * num_classes)
        logits = self.output_proj(input_sequence)
        # Reshape to (..., num_tickers, num_classes)
        logits = logits.reshape(logits.shape[:-1] + (self.num_tickers, self.num_classes))
        return logits 

    def train(self):
        """Set the model to training mode."""
        self._training = True
        # Set dropout layers to training mode
        self.input_dropout.train()
        for transformer_block in self.transformer_blocks:
            transformer_block.dropout.train()
            transformer_block.attention.train()

    def eval(self):
        """Set the model to evaluation mode."""
        self._training = False
        # Set dropout layers to evaluation mode
        self.input_dropout.eval()
        for transformer_block in self.transformer_blocks:
            transformer_block.dropout.eval()
            transformer_block.attention.eval()

    def apply_updates(self, grads):
        """Apply gradients to update model parameters using the nnx.Optimizer (in-place)."""
        self.optimizer.update(grads)
        return self 