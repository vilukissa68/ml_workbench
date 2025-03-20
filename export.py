#!/usr/bin/env python3
import torch
from quantize import is_model_quantized
from prune import is_model_pruned
from pathlib import Path
import os
from utils import load_model
from optimizers import get_optimizer


def export(model, dataset, args, quantized=False):
    model.to("cpu")
    if args.export_torchscript:
        path = (
            args.export_path
            + "/"
            + get_model_name(model, args, file_extension="_torchscript.pth")
        )
        export_torchscript(model, next(iter(dataset.get_example_input())), path)
    if args.export_torchdynamo:
        path = (
            args.export_path
            + "/"
            + get_model_name(model, args, file_extension="_torchdynamo.pth")
        )
        export_torchdynamo(model, path)
    if args.export_program:
        example_inputs = (next(iter(dataset.get_example_input())),)
        path = (
            args.export_path
            + "/"
            + get_model_name(model, args, file_extension="_exported.pt")
            )
        export_torch_program(model, example_inputs, path)
    if args.export_onnx and not quantized:
        path = (
            args.export_path
            + "/"
            + get_model_name(model, args, file_extension="_onnx.onnx")
        )
        export_onnx(model, next(iter(dataset.get_example_input())), path)



def export_torchscript(
    model, input_example, save_path="model_torchscript.pt", use_trace=True
):
    model.eval()  # Set to evaluation mode

    if use_trace:
        scripted_model = torch.jit.trace(model, input_example)
    else:
        scripted_model = torch.jit.script(model)

    scripted_model.save(save_path)
    print(f"TorchScript model saved to {save_path}")

    return scripted_model


def export_torchdynamo(model, save_path="model_torchdynamo.pth", backend="inductor"):
    model.eval()  # Set to evaluation mode

    compiled_model = torch.compile(model, backend=backend)

    # Save model weights (since compiled models can't be serialized directly)
    torch.save({"model_state_dict": compiled_model.state_dict()}, save_path)
    print(f"TorchDynamo model weights saved to {save_path}")

    return compiled_model


def export_torch_program(model, example_inputs, path):

    exported_program = torch.export.export(model, args=example_inputs)
    # Save the exported program to a file
    torch.save(exported_program, path)
    print(f"Exported program saved to {path}")
    return exported_program


def export_onnx(model, example_inputs, save_path="model.onnx"):
    model.eval()  # Set to evaluation mode

    # Export the model to ONNX format
    torch.onnx.export(
        model,
        example_inputs,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"ONNX model saved to {save_path}")


def load_exported_torch_program(path):
    return torch.load(path)


def save_checkpoint(
    model,
    args,
    optimizer,
    optimizer_name,
    epoch,
    model_name,
    dataset_name,
    batch_size,
    learning_rate,
    checkpoint_dir,
):
    # Ensure checkpoint directory exists
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    filename = get_model_name(
        model,
        args,
        model_name=model_name,
        dataset_name=dataset_name,
        batch_size=batch_size,
        epoch=epoch,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        file_extension=".pth",
    )
    checkpoint_path = checkpoint_dir / filename

    # Prepare checkpoint dictionary
    checkpoint = {
        "model_name": model_name,
        "dataset": dataset_name,
        "batch_size": batch_size,
        "lr": learning_rate,
        "epoch": epoch,
        "optimizer": optimizer_name,
        "quantized": is_model_quantized(model),
        "pruned": is_model_pruned(model),
        "model_state_dict": model.state_dict(),
    }

    # Only save optimizer state if it has parameters
    if optimizer.state_dict():
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)

    print(f"Checkpoint saved at: {checkpoint_path}")
    return str(checkpoint_path)


def get_model_name(
    model,
    args,
    model_name=None,
    dataset_name=None,
    batch_size=None,
    epoch=None,
    learning_rate=None,
    optimizer_name=None,
    file_extension=".pth",
):
    # Construct filename
    quantization_tag = "quantized" if is_model_quantized(model) else "fp32"
    pruning_tag = "pruned" if is_model_pruned(model) else "not_pruned"

    model_name = model_name if not model_name else args.model
    dataset_name = dataset_name if not dataset_name else args.dataset
    batch_size = batch_size if not batch_size else args.batch_size
    epoch = epoch if not epoch else int(args.epochs)
    learning_rate = learning_rate if not learning_rate else args.lr
    optimizer_name = optimizer_name if not optimizer_name else args.optimizer

    filename = f"{model_name}_{dataset_name}_{batch_size}_{learning_rate}_B_epoch{epoch}_{optimizer_name}_{quantization_tag}_{pruning_tag}_{args.timestamp}{file_extension}"
    return filename


def load_checkpoint(args, dataset):
    def load_statedict(model, optimizer, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        quantized = checkpoint["quantized"]
        pruned = checkpoint["pruned"]
        return model, optimizer, epoch, quantized, pruned

    _, file_extension = os.path.splitext(args.load_path)
    model_q = None
    model = None
    model_type = "pytorch"

    if args.verbose:
        print(f"Loading model from: {args.load_path}")

    # Tflite load
    if file_extension == ".tflite":
        print(args.load_path)
        tflite_model_buf = open(args.load_path, "rb").read()
        print("Loading TFLite model")
        model_type = "tflite"
        try:
            import tflite

            model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
        except AttributeError:
            import tflite.model

            model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    # Pytorch load
    else:
        model_name = torch.load(args.load_path)["model_name"]
        model = load_model(model_name, dataset.get_num_classes(), args)
        optimizer, _ = get_optimizer(model, args)
        model, optimizer, epoch, quantized, pruned = load_statedict(
            model, optimizer, args.load_path
        )
        if quantized:
            model_q = model
            model_q.to("cpu")
    return model, model_q, model_type
