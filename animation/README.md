# GPT Classifier Explainer Animation

This folder contains a Manim animation script to visualize the architecture and neuron activations of the GPTClassifier model used in the stock prediction pipeline.

## Files
- `gpt_classifier_explainer.py`: Main Manim script for the explainer animation.
- (Optional) `neuron_activations.json`: Data file with neuron activations for animation (see below).

## How to Run

1. **Install Manim** (if not already):
   ```bash
   pip install manim
   ```

2. **(Optional) Generate Neuron Activation Data:**
   - The animation will look for `../backtesting_results/neuron_activations.json`.
   - You can generate this file by modifying your pipeline to save activations, or create a mock file for demo:
     ```json
     {
       "layer1": [0.2, 0.5, 0.8],
       "layer2": [0.1, 0.6, 0.3],
       "layer3": [0.7, 0.4, 0.9]
     }
     ```

3. **Render the Animation:**
   ```bash
   manim -pql gpt_classifier_explainer.py GPTClassifierExplainer
   ```
   - `-pql` = preview, quick, low quality (for fast iteration)
   - For higher quality: `-pqh` or `-pqm`

4. **Output:**
   - The video will be saved in the `media/videos/animation/` directory by default.

## Customization
- To use real neuron activations, export them from your pipeline as a JSON file in the expected format.
- You can adjust the number of transformer blocks or the animation style in `gpt_classifier_explainer.py`.

---

For more info on Manim: [https://docs.manim.community/](https://docs.manim.community/) 