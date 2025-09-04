import json
import os

# Example mock activations for 3 transformer blocks, 3 neurons each
mock_activations = {
    "layer1": [0.2, 0.5, 0.8],
    "layer2": [0.1, 0.6, 0.3],
    "layer3": [0.7, 0.4, 0.9]
}

output_path = os.path.join(os.path.dirname(__file__), '../backtesting_results/neuron_activations.json')
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(mock_activations, f, indent=2)

print(f"Mock neuron activations written to {output_path}") 