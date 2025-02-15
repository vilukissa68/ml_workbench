#!/usr/bin/env python3

from train import train, load_checkpoint, get_optimizer
import models
import args_parser
import torchvision.transforms as transforms
from benchmark_inference import benchmark_inference  # Import the benchmarking function
from visualize import imshow_batch
import torch
import os
from quantize import ptsq
from qat import train_qat
from prune import prune_model_global
from datasets import mnist, cifar10, mlperf_tiny_kws
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def load_model(model_name, num_classes, args):
    # Dynamically load the model based on the argument
    model_func = getattr(models, model_name, None)  # Find model function by name
    if model_func is None:
        raise ValueError(f"Model '{model_name}' is not defined in models/resnet.py")

    # Load the model using the selected function
    model = model_func(num_classes=num_classes, args=args)
    return model


def get_dataset(dataset_name, transform):
    dataset_name = dataset_name.lower()
    if dataset_name == "cifar10":
        return cifar10.CIFAR10(transform=transform)
    elif dataset_name == "mnist":
        return mnist.MNIST(transform=transform)
    elif dataset_name == "mlperftinykws":
        return mlperf_tiny_kws.MLPerfTinyKWS(transform=transform)
    else:
        raise Exception(f"No dataset named {dataset_name} found.")


def run_benchmark(args, dataset, model=None, model_q=None):
    if args.benchmark:
        if model:
            print("Running inference benchmarking...")
            benchmark_inference(
                model,
                dataset,
                args.batch_size,
                args.device,
                num_iterations=args.benchmark_num_iterations,
            )
        if model_q:
            print("Running quantized inference benchmark...")
            benchmark_inference(
                model_q,
                dataset,
                args.batch_size,
                "cpu",
                num_iterations=args.benchmark_num_iterations,
            )


def load_checkpoint(args):
    _, file_extension = os.path.splitext(args.load_checkpoint_path)
    model_q = None
    model = None
    model_type = "pytorch"

    # Tflite load
    if file_extension == ".tflite":
        print(args.load_checkpoint_path)
        tflite_model_buf = open(args.load_checkpoint_path, "rb").read()
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
        model_name = torch.load(args.load_checkpoint_path)["model_name"]
        model = load_model(model_name, num_classes, args)
        optimizer = get_optimizer(args.optimizer, model, args.lr)
        model, optimizer, epoch, quantized, pruned = load_checkpoint(
            model, optimizer, args.load_checkpoint_path
        )
        if quantized:
            model_q = model
            model_q.to("cpu")
    return model, model_q, model_type


def main():
    args = args_parser.parse_args()

    print(f"Arguments: {args}")

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    dataset = get_dataset(args.dataset, transform)

    writer = None
    if args.tensorboard:
        print("Tensorboard loggin enabled.")
        print("Make sure tensorboard is running: tensorboard --logdir=runs")
        timestamp = datetime.now().strftime("%y-%m-%d-%H-%M")
        run_name = f"{args.model}_{args.dataset}_{args.batch_size}_{timestamp}"
        writer = SummaryWriter(f"runs/{run_name}")

    if args.load_checkpoint_path:
        model, model_q, model_type = load_checkpoint(args)
    else:  # Load empty model
        model = load_model(args.model, dataset.get_num_classes(), args)
        model_q = None

    if args.train or args.fine_tune or args.qat:
        if args.train:
            print("Training model...")
        elif args.fine_tune:
            print("Fine tuning model...")

        if args.qat:
            print("Quantization aware training...")
            model = train_qat(model, dataset, args)
            q_model = model
        else:
            train(model, dataset, args, writer)
        model_type = "pytorch"

    # Quantizie fp32, skip if using QAT already
    if args.quantize and not args.qat:
        if args.quantization_method == "ptsq":
            model_q = ptsq(model, dataset, wrap=False)
            model = None
        else:
            print("Error! Unknown quantization method.")
            return
        run_benchmark(args, dataset, model, model_q)

    if args.prune:
        if model_q:
            q_model = prune_model_global(model_q)
        else:
            model = prune_model_global(model)
        run_benchmark(args, dataset, model, model_q)

    if args.visualize:
        images, labels = next(iter(test_loader))
        imshow_batch(images, labels=labels, normalize=True)

    if args.tvm_export:
        from integrations import tvm_export_model, tvm_compile

        mod, params = tvm_export_model(model, dataset.get_data_shapes(), model_type)
        tvm_compile(mod, params, "./tvm_builds")
        print("TVM compilation finished.")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
