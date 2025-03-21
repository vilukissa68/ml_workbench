#!/usr/bin/env python3
import torch
import torch.quantization
from torch.multiprocessing import Pool, Manager
from tqdm import tqdm
import math
import torch.nn as nn


class QuantWrapper(torch.nn.Module):
    def __init__(self, model):
        super(QuantWrapper, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.model = model
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


def calibrate(model, data_loader, num_batches=10, device="cpu"):
    """Runs calibration with tqdm progress bar."""
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        for i, (inputs, _) in enumerate(
            tqdm(data_loader, total=num_batches, desc="Calibrating")
        ):
            inputs = inputs.to(device)  # Move data to the correct device
            model(inputs)  # Forward pass only

            if i >= num_batches - 1:
                break  # Stop after the specified number of batches


def fuse_model_layers(model):
    """
    Automatically fuses common layer patterns in a PyTorch model for quantization.

    Fuses:
    - (Conv2d, ReLU) → Conv2d
    - (Linear, ReLU) → Linear
    - (Conv2d, BatchNorm2d, ReLU) → Conv2d
    - (Conv2d, BatchNorm2d) → Conv2d

    Args:
        model (torch.nn.Module): The model to fuse.

    Returns:
        None (modifies model in place).
    """
    for module_name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            # Recursively fuse modules inside Sequential blocks
            fuse_model_layers(module)

        # Define fusable layer patterns
        fusion_list = []
        if isinstance(module, nn.Conv2d):
            next_module = getattr(model, module_name + "_next", None)
            if isinstance(next_module, nn.ReLU):
                fusion_list = [module_name, module_name + "_next"]
        elif isinstance(module, nn.BatchNorm2d):
            prev_module_name = module_name.replace("_next", "")
            prev_module = getattr(model, prev_module_name, None)
            if isinstance(prev_module, nn.Conv2d):
                fusion_list = [prev_module_name, module_name]
        elif isinstance(module, nn.Linear):
            next_module = getattr(model, module_name + "_next", None)
            if isinstance(next_module, nn.ReLU):
                fusion_list = [module_name, module_name + "_next"]

        # Perform fusion if a pattern is found
        if fusion_list:
            model = torch.quantization.fuse_modules(model, fusion_list, inplace=True)
        return model


def ptsq(
    model,
    dataset,
    args,
    wrap=True,
    calibration_batches=100,
):

    # Prepare data for calibration
    dataset.load_data(["val"])
    _, _, data_loader = dataset.get_data_loaders()

    # Model needs to be quantized on CPU
    model.to("cpu")

    # Setup quantization engine
    torch.backends.quantized.engine = args.quantization_backend

    # Wrap model with fake quants
    if wrap:
        print("Wrapping model with fake quantization layers")
        model = QuantWrapper(model)

    model.eval()
    # model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
    model.qconfig = get_quantization_scheme(
        args.quantization_scheme, args.quantization_backend
    )

    model_fused = fuse_model_layers(model)
    model_prepared = torch.ao.quantization.prepare(model, inplace=False)

    calibrate(
        model_prepared,
        data_loader,
        calibration_batches,
    )

    model_q = torch.ao.quantization.convert(model_prepared, inplace=False)
    if args.verbose:
        print("Quantized model:", model_q)

    return model_q


def is_model_quantized(model):
    """
    Check if the model is quantized.

    Args:
    - model (torch.nn.Module): The model to check.

    Returns:
    - bool: True if the model is quantized, False otherwise.
    """
    # Check if the model has quantized layers (Conv2d, Linear, etc.)
    for module in model.modules():
        if isinstance(module, (torch.nn.quantized.Conv2d, torch.nn.quantized.Linear)):
            return True

    # Check for quantization stubs (QuantStub and DeQuantStub) inserted during quantization
    if isinstance(model, torch.quantization.QuantStub):
        return True

    return False


def get_quantization_scheme(scheme, backend="qnnpack"):
    if scheme == "int8-symmetric":
        return torch.quantization.QConfig(
            activation=torch.quantization.FakeQuantize.with_args(
                observer=torch.quantization.MinMaxObserver,
                quant_min=-128,
                quant_max=127,
                qscheme=torch.per_tensor_symmetric,
                dtype=torch.qint8,
                reduce_range=False,
            ),
            weight=torch.quantization.FakeQuantize.with_args(
                observer=torch.quantization.MinMaxObserver,
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric,
                reduce_range=False,
            ),
        )
    else:
        # Default to backend specific qconfig
        return torch.quantization.get_default_qconfig(backend)
