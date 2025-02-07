import torch
import torch.nn.utils.prune as prune
import torch.nn as nn


def prune_model_global(model, amount=0.5, pruning_method=prune.L1Unstructured):
    """
    Automatically prunes all Conv2d and Linear layers in a model using global pruning.

    Args:
        model (torch.nn.Module): The PyTorch model to prune.
        amount (float): Fraction of weights to prune globally.
        pruning_method (prune.BasePruningMethod): The pruning method (default is L1 unstructured).

    Returns:
        None (modifies the model in place).
    """
    parameters_to_prune = []

    # Automatically find all Conv2d and Linear layers
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, "weight"))  # Add weight parameter

    if len(parameters_to_prune) == 0:
        print("No prunable layers found.")
        return

    # Apply global unstructured pruning
    prune.global_unstructured(
        parameters_to_prune, pruning_method=pruning_method, amount=amount
    )

    print(
        f"Pruned {amount * 100}% of the weights globally across {len(parameters_to_prune)} layers."
    )
