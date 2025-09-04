import os
import sys
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from manim import *
import numpy as np
import json
from nnx_model_structure import extract_nnx_structure
from src.models.gpt_classifier import GPTClassifier
from neural_network_manim import NeuralNetworkMobject
from src.scripts import backtesting

# ---
# This file provides two visualization classes:
# 1. NNXModelBlockDiagram: Faithful block/arrow diagram (no ANN/neuron visuals)
# 2. NNXModelANNDiagram: Uses NeuralNetworkMobject for neuron-style ANN visuals, colored/labeled by type
# Both use extract_nnx_structure for true structure. Use either as the scene class in Manim.
# ---

# Color mapping for layer types
LAYER_COLORS = {
    'Linear': YELLOW,
    'Dropout': ORANGE,
    'MultiHeadAttention': PURPLE,
    'LayerNorm': TEAL,
    'LIF': RED,
    'PositionalEncoding': GREEN,
    'GPTClassifier': BLUE,
    'TransformerBlock': BLUE,
}

# Icon mapping for layer types (optional, can use Text or MathTex)
LAYER_ICONS = {
    'Linear': 'L',
    'Dropout': 'D',
    'MultiHeadAttention': 'A',
    'LayerNorm': 'N',
    'LIF': '\u03BB',  # lambda for spiking
    'PositionalEncoding': 'PE',
    'GPTClassifier': 'G',
    'TransformerBlock': 'T',
}

class NNXModelBlockDiagram(Scene):
    """
    Faithful block/arrow diagram of the model structure (no ANN/neuron visuals).
    Each block = real submodule/layer, colored/labeled by type, with arrows for data flow.
    Now with neuron circles inside each block, colored by type.
    """
    def construct(self):
        be = backtesting.BacktestingEngine()
        model = be.model
        if model is None:
            model = GPTClassifier(
                num_layers=2,
                d_model=16,
                num_heads=2,
                d_ff=32,
                num_classes=3,
                dropout_rate=0.1,
                input_features=4,
                num_tickers=2,
                maxlen=10
            )

        layers = extract_nnx_structure(model)
        print("[DEBUG] Extracted layers:", layers)
        vis_layers = []
        for layer in layers:
            # Determine neuron count from shape
            size = None
            shape = layer.get("shape", None)
            if isinstance(shape, tuple) and len(shape) > 0:
                # For LIF, use the last dimension if shape is (1, N) or (B, N)
                if layer["type"] == "LIF" and len(shape) > 1:
                    size = shape[-1]
                else:
                    size = shape[-1]
            elif isinstance(shape, int):
                size = shape
            if size is not None and size > 0:
                vis_layers.append({
                    "label": f"{layer['name'].split('.')[-1]}\n{layer['type']}\n{layer['shape']}",
                    "type": layer["type"],
                    "size": size
                })
        print("[DEBUG] vis_layers:", vis_layers)
        if not vis_layers:
            vis_layers = [{"label": "Model", "type": "GPTClassifier", "size": 4}]
        x_start = -len(vis_layers) // 2 * 2.5
        y = 0
        block_width = 2.2
        block_height = 1.2
        blocks = []
        block_mobs = []
        all_mobjects = VGroup()
        for i, layer in enumerate(vis_layers):
            x = x_start + i * 2.5
            color = LAYER_COLORS.get(layer["type"], GREY)
            rect = Rectangle(width=block_width, height=block_height, color=color, fill_opacity=0.2).move_to([x, y, 0])
            icon = LAYER_ICONS.get(layer["type"], '?')
            icon_text = Text(icon, font_size=28, color=color).next_to(rect, UP, buff=0.1)
            text = Text(layer["label"], font_size=16).move_to(rect.get_center())
            # Add neurons inside block
            n_neurons = max(2, min(layer["size"], 12))  # Always at least 2 neurons
            neuron_radius = 0.13
            neuron_color = color
            neurons = VGroup()
            # Center neurons vertically in the block
            total_height = (n_neurons-1)*neuron_radius*1.5
            y0 = rect.get_center()[1] + total_height/2
            for j in range(n_neurons):
                neuron = Circle(radius=neuron_radius, stroke_color=neuron_color, fill_color=neuron_color, fill_opacity=0.7, stroke_width=2)
                neuron.move_to([x, y0 - j*neuron_radius*1.5, 0])
                neurons.add(neuron)
            all_mobjects.add(rect, icon_text, text, neurons)
            blocks.append(rect)
            block_mobs.append((rect, icon_text, text, neurons))
        arrow_mobs = VGroup()
        for i in range(len(blocks) - 1):
            arrow = Arrow(blocks[i].get_right(), blocks[i+1].get_left(), buff=0.1)
            arrow_mobs.add(arrow)
        all_mobjects.add(arrow_mobs)
        # Fit to screen after all mobjects are created
        margin = 1.0  # units to leave as margin
        all_mobjects.scale_to_fit_width(config.frame_width - margin)
        if all_mobjects.height > (config.frame_height - margin):
            all_mobjects.scale_to_fit_height(config.frame_height - margin)
        all_mobjects.move_to(ORIGIN)
        self.add(all_mobjects)
        # Animate after fitting to screen
        for i, (rect, icon_text, text, neurons) in enumerate(block_mobs):
            self.play(
                rect.animate.set_fill(LAYER_COLORS.get(vis_layers[i]["type"], GREY), opacity=0.7),
                *[n.animate.set_fill(LAYER_COLORS.get(vis_layers[i]["type"], GREY), opacity=1.0) for n in neurons],
                run_time=0.5)
            self.wait(0.3)
            self.play(
                rect.animate.set_fill(LAYER_COLORS.get(vis_layers[i]["type"], GREY), opacity=0.2),
                *[n.animate.set_fill(LAYER_COLORS.get(vis_layers[i]["type"], GREY), opacity=0.7) for n in neurons],
                run_time=0.2)
        self.wait(1)
        # If you see no blocks or neurons, check the [DEBUG] output above for what layers are being parsed.

class NNXModelANNDiagram(Scene):
    """
    Uses NeuralNetworkMobject for neuron-style ANN visuals, colored/labeled by type.
    Each layer = real submodule/layer, colored/labeled by type, with neuron visuals and edges.
    """
    def construct(self):
        # Create a demo model instance (adjust params as needed)
        be = backtesting.BacktestingEngine()
        model = be.model
        if model is None:
            model = GPTClassifier(
                num_layers=2,
                d_model=16,
                num_heads=2,
                d_ff=32,
                num_classes=3,
                dropout_rate=0.1,
                input_features=4,
                num_tickers=2,
                maxlen=10
            )
        # Extract structure
        layers = extract_nnx_structure(model)
        # Build layer sizes/types for NeuralNetworkMobject
        ann_layers = []
        ann_types = []
        ann_labels = []
        for layer in layers:
            size = None
            shape = layer.get("shape", None)
            if isinstance(shape, tuple) and len(shape) > 0:
                if layer["type"] == "LIF" and len(shape) > 1:
                    size = shape[-1]
                else:
                    size = shape[-1]
            elif isinstance(shape, int):
                size = shape
            if size is not None and size > 0:
                ann_layers.append(size)
                ann_types.append(layer["type"])
                ann_labels.append(f"{layer['name'].split('.')[-1]}\n{layer['type']}\n{layer['shape']}")
        if not ann_layers:
            ann_layers = [1]
            ann_types = ["GPTClassifier"]
            ann_labels = ["Model"]
        # Create the ANN mobject
        ann = NeuralNetworkMobject(ann_layers)
        ann.move_to(ORIGIN)
        self.add(ann)
        # Overlay custom labels and color each layer
        for i, (layer_type, label) in enumerate(zip(ann_types, ann_labels)):
            color = LAYER_COLORS.get(layer_type, GREY)
            icon = LAYER_ICONS.get(layer_type, '?')
            # Color neurons in this layer
            for neuron in ann.layers[i].neurons:
                neuron.set_stroke(color, width=3)
                neuron.set_fill(color, opacity=0.3)
            # Add icon and label above layer
            icon_text = Text(icon, font_size=28, color=color).next_to(ann.layers[i], UP, buff=0.1)
            label_text = Text(label, font_size=16).next_to(icon_text, UP, buff=0.05)
            self.add(icon_text, label_text)
        # Animate activations: highlight each layer in sequence
        for i, layer in enumerate(ann.layers):
            self.play(*[n.animate.set_fill(LAYER_COLORS.get(ann_types[i], GREY), opacity=0.9) for n in layer.neurons], run_time=0.5)
            self.wait(0.3)
            self.play(*[n.animate.set_fill(LAYER_COLORS.get(ann_types[i], GREY), opacity=0.3) for n in layer.neurons], run_time=0.2)
        self.wait(1)

# To use: manim animation/gpt_classifier_explainer.py NNXModelBlockDiagram
#      or: manim animation/gpt_classifier_explainer.py NNXModelANNDiagram

if __name__ == "__main__":
    from manim import cli
    cli.main() 