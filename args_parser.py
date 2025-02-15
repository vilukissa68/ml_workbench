import argparse
import inspect
import models
import datasets


# Extract all function names from the models module
def get_available_models():
    return [name for name, func in inspect.getmembers(models, inspect.isfunction)]


def get_available_datasets():
    return [name for name, func in inspect.getmembers(datasets, inspect.isclass)]


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Workbench for experiments")

    # Run flags
    parser.add_argument(
        "--train",
        action="store_true",
        help="Flag to trigger training. If not set, will only benchmark.",
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Flag to trigger benchmarking."
    )
    parser.add_argument("--prune", action="store_true", help="Flag to prune the model.")
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize part of the dataset."
    )
    parser.add_argument(
        "--tensorboard", action="store_true", help="Visualize training with Tensorboard"
    )

    # General options
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda", "mps"],
        help="Device to run on, either 'cpu', 'cuda' or 'mps'. Default is 'cuda'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default is 42.",
    )

    # Model options
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=get_available_models(),
        help="Model architecture to use. Default is 'resnet18'.",
    )

    # Data options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training. Default is 32.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for training. Default is 10.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        choices=get_available_datasets(),
        help="Dataset to use for training. Default is 'CIFAR10'.",
    )

    # Optimizer options
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        choices=["SGD", "Adam", "RMSprop"],
        help="Optimizer to use. Default is 'SGD'.",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate. Default is 0.001."
    )

    # Quantization options
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Flag to enable model quantization",
    )
    parser.add_argument(
        "--save-quantized",
        action="store_true",
        help="Save quantized checkpoints. Only applies to PTQ.",
    )

    parser.add_argument(
        "--qat",
        action="store_true",
        help="Flag to use quantization aware training.",
    )
    parser.add_argument(
        "--quantization-method",
        type=str,
        default="ptsq",
        choices=["ptsq"],
        help="Define methods used for quantization.",
    )
    parser.add_argument(
        "--quantization-backend",
        type=str,
        default="qnnpack",
        choices=["qnnpack", "x86", "fbgemm"],
        help="Define quantization backend.",
    )

    # Checkpoint options
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Path to save model checkpoints. If None, no checkpoints are saved.",
    )

    # Checkpoint options
    parser.add_argument(
        "--load-checkpoint-path",
        type=str,
        default=None,
        help="Path to model checkpoint to load. If empty no checkpoint will be loaded.",
    )

    # Logging options
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory to save logs. Default is './logs'.",
    )

    # Benchmarking options
    parser.add_argument(
        "--benchmark-num-iterations",
        type=int,
        default=100,
        help="Number of iterations during inference benchmarking.",
    )

    # Fine tuning
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Load pretrained version of torchvision models for finetuning",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Pretrained weigths for torchvision models",
    )

    # Other options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, enables verbose output during training.",
    )

    # TVM integration options
    parser.add_argument(
        "--tvm-export",
        action="store_true",
        help="Flag to export model to TVM.",
    )
    parser.add_argument(
        "--tvm-export-quantized",
        action="store_true",
        help="Flag to export quantized model to TVM.",
    )

    return parser.parse_args()
