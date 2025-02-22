import math
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def plot_confusion_matrix_to_tensorboard(
    y_true, y_pred, class_names, writer, global_step
):
    """
    Plots a confusion matrix and logs it to TensorBoard.

    Args:
    - y_true (list or torch.Tensor): Ground truth labels.
    - y_pred (list or torch.Tensor): Predicted labels.
    - class_names (list): List of class names to label the axes.
    - writer (SummaryWriter): TensorBoard writer to log the image.
    - global_step (int): Global step or epoch to associate the log entry with.
    """
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize the confusion matrix (optional)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create a figure to plot the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    # Set labels
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix (Normalized)")

    # Convert the figure to a PIL Image
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")  # Save the figure as a PNG file
    plt.close(fig)  # Close the figure to avoid display

    # Open the saved image
    img = Image.open("confusion_matrix.png")

    # Convert image to Tensor
    transform = transforms.ToTensor()
    img_tensor = transform(img)  # Add batch dimension

    # Log the image to TensorBoard
    writer.add_image("Confusion Matrix", img_tensor, global_step)


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
