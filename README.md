# UNet Polygon

This readme details the implementation, training, and challenges of creating a Conditional UNet to color polygon shapes. The model takes a binary polygon image and a color name as input and generates an image of the polygon filled with the specified color.

## 1. Final Hyperparameters and Configuration

After experimentation and debugging, the final, stable configuration was determined. The model was trained on a Google Colab T4 GPU.

| Hyperparameter      | Final Value | Rationale                                                                                             |
|---------------------|-------------|-------------------------------------------------------------------------------------------------------|
| **Image Size**      | 128x128     | Provides a good balance between visual detail and computational/VRAM cost. Edges are sufficiently sharp at this resolution. |
| **Batch Size**      | 16          | A standard size that fit comfortably in the T4 GPU's VRAM for a 128x128 image size.                     |
| **Learning Rate**   | 1e-3        | Adam optimizer with this learning rate showed fast and stable convergence.                               |
| **Epochs**          | 75          | The validation loss plateaued around epoch 60-70. 75 epochs were sufficient to ensure convergence without significant overfitting. |
| **Loss Function**   | MSELoss     | Mean Squared Error provided a strong signal for pixel-wise reconstruction, resulting in clean, accurate fills. |
| **Color Embed Dim** | 32          | With only 8 unique colors, an embedding dimension of 32 provides enough capacity for the model to distinguish color signals without being wasteful. |

## 2. Model Architecture

The core of the model is a standard **UNet architecture** implemented from scratch in PyTorch.

- **Encoder:** Consists of 4 down-sampling blocks. Each block uses a `MaxPool2d` followed by a `DoubleConv` module (`Conv -> BN -> ReLU` repeated twice). This path extracts hierarchical features from the input polygon shape.
- **Bottleneck:** A `DoubleConv` block at the lowest resolution (1/16th of the input size).
- **Decoder:** Consists of 4 up-sampling blocks that mirror the encoder. Each block uses bilinear upsampling (`Upsample`) to increase resolution, followed by a concatenation with the corresponding high-resolution feature map from the encoder's skip connection. This allows the decoder to reconstruct fine-grained details.

### Conditioning Strategy

The color information is injected into the model during the decoding process. This is a powerful technique that allows the model to modify its generative process based on the color condition.

1.  **Color Embedding:** The input color name (e.g., "red") is mapped to a unique integer index. This index is passed to a `torch.nn.Embedding` layer, which produces a dense vector of size 32.

2.  **Additive Injection in Decoder:** At each of the 4 up-sampling stages of the decoder, the 32-dimensional color embedding is projected by a small MLP (a single `nn.Linear` layer) to match the number of channels of the current feature map. This projected vector is then broadcasted and **added** to the feature map right after the skip connection is concatenated.

This additive conditioning acts like a "style" signal, biasing the activations of the decoder to produce the desired color output at multiple resolutions.

## 3. Training Dynamics and Key Learnings

The project's journey from setup to a working model was highly instructive, with data handling being the most significant challenge.

### Training Performance
The final training run was very successful. As seen in the W&B logs, the training and validation loss curves show healthy, rapid convergence. The validation loss decreased steadily from ~0.27 to ~0.0004, indicating the model learned the task extremely well.

 <!-- It's highly recommended to replace this with a screenshot of your actual loss curve from W&B -->

### Qualitative Results
Visual inspection of the outputs (logged to WandB) confirmed the model's success. It accurately segments the polygon from the background and fills it with the correctly specified color. The model generalizes to different shapes and colors seamlessly.

 <!-- It's highly recommended to replace this with a screenshot of your sample outputs grid from W&B -->

### Key Learnings & Debugging Journey

The most critical lessons came from debugging the data loading pipeline.

1.  **Data Format is King (`KeyError`)**: The initial code failed due to a mismatch between the assumed keys (`'input'`, `'color'`) and the actual keys in the `data.json` files (`'input_polygon'`, `'colour'`). **Learning:** *Never trust documentation blindly; always programmatically inspect a few samples of your data before writing the data loader.* A small debug script that printed the first item of the JSON was essential for solving this.

2.  **Inconsistent Vocabularies (`AssertionError`)**: The training and validation sets contained a different number of unique colors. The initial `Dataset` class created a vocabulary based only on the data it saw, leading to two different color-to-index mappings. This would have caused silent failures during validation. **Learning:** *Training and validation data must be processed with a single, unified pipeline.* The fix was to create a "master vocabulary" from all available data *before* initializing the `Dataset` objects and passing this unified vocabulary to them.

3.  **The Power of `wandb`**: Experiment tracking was invaluable.https://wandb.ai/lancelot03-chandigarh-university/polygon-coloring-unet?nw=nwuserlancelot03
    *   **Logging Hyperparameters:** Made it easy to track which settings led to which results.
    *   **Visualizing Metrics:** The live loss curves immediately showed if a model was learning, diverging, or overfitting.
    *   **Visualizing Outputs:** Logging image predictions at each epoch was the best way to understand *what* the model was learning. Seeing blurry gray blobs evolve into sharp, correctly colored polygons provided more insight than any loss value alone.

### Typical Failure Modes
- **Initial Blurriness:** In early epochs, the model produced blurry, grayish outputs. This is typical for pixel-wise regression losses like MSE as the model first learns the general shape and "average" color before refining details.
- **Color Bleeding:** Very early on, some color would "bleed" into the background. This was quickly corrected by the MSE loss, which heavily penalizes any deviation from the pure white background (pixel value 1.0).

## 4. Conclusion

This project successfully demonstrated the implementation of a conditional UNet for a generative image-to-image task. While the model architecture is relatively standard, the project underscored the critical importance of robust data validation and a unified data processing pipeline. The final model performs its task accurately and efficiently, serving as an excellent case study in conditional image generation.
