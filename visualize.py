import math
import torch
import matplotlib.pyplot as plt
import numpy as np


def imshow_batch(images, labels=None, normalize=False):
    """
    Displays a batch of images in a grid.

    Args:
        images (torch.Tensor): A batch of images with shape (B, C, H, W).
        labels (list or torch.Tensor, optional): Corresponding labels.
        normalize (bool): Whether to denormalize images (if they were normalized during preprocessing).
    """
    batch_size = images.shape[0]

    # Determine grid size (rows, cols)
    cols = math.ceil(math.sqrt(batch_size))
    rows = math.ceil(batch_size / cols)

    # Convert images to NumPy
    images = images.numpy().transpose(
        (0, 2, 3, 1)
    )  # Change from (B, C, H, W) to (B, H, W, C)

    if normalize:
        images = (images * 0.5) + 0.5  # Assuming normalization was mean=0.5, std=0.5

    # Plot images
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() if batch_size > 1 else [axes]  # Handle single image case

    for i in range(batch_size):
        axes[i].imshow(images[i])
        axes[i].axis("off")
        if labels is not None:
            axes[i].set_title(f"{labels[i]}")

    # Hide any unused subplots
    for i in range(batch_size, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
