#!/usr/bin/env python3

import argparse


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
    parser.add_argument(
        "--quantize", action="store_true", help="Flag to quantize the model."
    )
    parser.add_argument("--prune", action="store_true", help="Flag to prune the model.")

    # General options
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to run on, either 'cpu' or 'cuda'. Default is 'cuda'.",
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
        choices=["resnet18", "resnet34", "resnet50"],
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
        choices=["CIFAR10", "MNIST", "ImageNet"],
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
    # Checkpoint options
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Path to save model checkpoints. If None, no checkpoints are saved.",
    )

    # Logging options
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory to save logs. Default is './logs'.",
    )

    # Other options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, enables verbose output during training.",
    )

    return parser.parse_args()
