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
    # Loss options
    parser.add_argument(
        "--criterion",
        type=str,
        default="CrossEntropyLoss",
        choices=[
            "CrossEntropyLoss",
            "MSELoss",
            "L1Loss",
            "NLLLoss",
            "BCELoss",
            "BCEWithLogitsLoss",
        ],
        help="Critetion/Loss function used for training.",
    )

    # Optimizer options
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        choices=["SGD", "Adam", "AdamW", "RMSprop"],
        help="Optimizer to use. Default is 'SGD'.",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate. Default is 0.001."
    )

    # Scheduler options
    parser.add_argument(
        "--scheduler",
        type=str,
        default="",
        choices=["StepLR", "MultiStepLR", "ReduceLROnPlateau", "LambdaLR"],
        help="Scheduler to use. Default is 'StepLR'.",
    )

    parser.add_argument(
        "--scheduler-step-size",
        type=int,
        default=12,
        help="Step size for the scheduler. Default is 7.",
    )

    parser.add_argument(
        "--scheduler-gamma",
        type=float,
        default=0.000005,
        help="Gamma for the scheduler. Default is 0.1.",
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

    parser.add_argument(
        "--quantization-scheme",
        type=str,
        default="",
        choices=["int8-symmetric"],
        help="Define quantization scheme. Sets QConfig. If empty, default to backend QConfig.",
    )

    # Checkpoint options
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Path to save model checkpoints. If None, no checkpoints are saved.",
    )
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

    # Export options
    parser.add_argument(
        "--export_path",
        type=str,
        default="./checkpoints",
        help="Path to save exported models.",
    )
    parser.add_argument(
        "--export-torchscript",
        action="store_true",
        help="Flag to export loaded model in torchscript format.",
    )
    parser.add_argument(
        "--export-torchdynamo",
        action="store_true",
        help="Flag to export loaded model in torchdynamo format.",
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

    # Distributed training
    parser.add_argument(
        "--distributed-training",
        action="store_true",
        help="Use DataParallelDistributed training.",
    )
    parser.add_argument(
        "--ngpus",
        type=int,
        default=1,
        help="Number of GPUs per node in DPP training",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Total number of nodes per in DPP training",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="Local rank of training node in DPP training. Should be set separately for all nodes [0, num_of_nodes-1].",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Sets number of parallel workers in data loaders",
    )
    parser.add_argument(
        "--ip_adress", type=str, default="127.0.0.1", help="ip address of the host node"
    )

    # Training regularization
    parser.add_argument(
        "--reg-lambda",
        type=float,
        default=0.01,
        help="Lambda for regularization.",
    )
    parser.add_argument(
        "--regularization",
        type=str,
        default="",
        choices=["L1", "L2"],
        help="Lambda for regularization.",
    )
    # Optimizer parameters
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for optimizer.",
    )
    parser.add_argument(
        "--optimizer-beta1",
        type=float,
        default=None,
        help="Low coefficient for optimizer beta",
    )
    parser.add_argument(
        "--optimizer-beta2",
        type=float,
        default=None,
        help="High coefficient for optimizer beta",
    )

    # Plotting options
    parser.add_argument(
        "--plot-color-palette",
        type=str,
        default="magma",
        help="Color palette to use for plotting.",
    )
    parser.add_argument(
        "--plot-style",
        type=str,
        default="whitegrid",
        choices=["whitegrid", "darkgrid", "white", "dark", "ticks"],
        help="Style to use for plotting.",
    )
    return parser.parse_args()
