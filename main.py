#!/usr/bin/env python3

from train import train, get_optimizer
from export import export, load_checkpoint
import models
import args_parser
from benchmark_inference import benchmark_inference  # Import the benchmarking function
from visualize import imshow_batch, plot_confusion_matrix_to_tensorboard
import torch
import os
from quantize import ptsq
from qat import train_qat
from prune import prune_model_global
from datasets import mnist, cifar10, mlperf_tiny_kws
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist

torch.multiprocessing.set_sharing_strategy("file_system")

TIMESTAMP = datetime.now().strftime("%y-%m-%d-%H-%M")


def load_model(model_name, num_classes, args):
    # Dynamically load the model based on the argument
    model_func = getattr(models, model_name, None)  # Find model function by name
    if model_func is None:
        raise ValueError(f"Model '{model_name}' is not defined in models/resnet.py")

    # Load the model using the selected function
    model = model_func(num_classes=num_classes, args=args)
    return model


def get_dataset(dataset_name, args):
    dataset_name = dataset_name.lower()
    if dataset_name == "cifar10":
        return cifar10.CIFAR10()
    elif dataset_name == "mnist":
        return mnist.MNIST()
    elif dataset_name == "mlperftinykws":
        return mlperf_tiny_kws.MLPerfTinyKWS()
    else:
        raise Exception(f"No dataset named {dataset_name} found.")


def run_benchmark(args, dataset, model=None, model_q=None):
    return benchmark_inference(
        model,
        dataset,
        args.batch_size,
        args.device,
        num_iterations=args.benchmark_num_iterations,
    )
    benchmark_inference(
        model_q,
        dataset,
        args.batch_size,
        "cpu",
        num_iterations=args.benchmark_num_iterations,
    )


def main():
    args = args_parser.parse_args()

    print(f"Arguments: {args}")
    args.timestamp = TIMESTAMP

    writer = None
    if args.tensorboard:
        print("Tensorboard loggin enabled.")
        print("Make sure tensorboard is running: tensorboard --logdir=runs")
        run_name = f"{args.model}_{args.dataset}_{args.batch_size}_{TIMESTAMP}"
        writer = SummaryWriter(f"runs/{run_name}")

    # Adjust batch_size for distributed training
    args.batch_size = int(args.batch_size / args.ngpus)
    dataset = get_dataset(args.dataset, args)

    if args.load_checkpoint_path:
        model, model_q, model_type = load_checkpoint(args)
    else:  # Load empty model
        model = load_model(args.model, dataset.get_num_classes(), args)
        model_q = None

    if args.train or args.fine_tune or args.qat:
        if args.qat:
            print("Quantization aware training...")
            model = train_qat(model, dataset, args)
            q_model = model
        else:
            if args.train:
                print("Training model...")
            elif args.fine_tune:
                print("Fine tuning model...")
            if args.distributed_training:
                args.world_size = args.ngpus * args.nodes
                # add the ip address to the environment variable so it can be easily avialbale
                os.environ["MASTER_ADDR"] = args.ip_adress
                print("ip_adress is", args.ip_adress)
                os.environ["MASTER_PORT"] = "8888"
                os.environ["WORLD_SIZE"] = str(args.world_size)
                # nprocs: number of process which is equal to args.ngpu here
                mp.spawn(
                    train, nprocs=args.ngpus, args=(model, dataset, args), join=True
                )
            else:
                train(0, model, dataset, args, writer)
        model_type = "pytorch"

    # Quantizie fp32, skip if using QAT already
    if args.quantize and not args.qat:
        if args.quantization_method == "ptsq":
            model_q = ptsq(model, dataset, wrap=False)
            model = None
        else:
            print("Error! Unknown quantization method.")
            return
        run_benchmark(args, dataset, model, model_q, writer)

    if args.prune:
        if model_q:
            q_model = prune_model_global(model_q)
        else:
            model = prune_model_global(model)
        run_benchmark(args, dataset, model, model_q)

    if args.visualize:
        images, labels = next(iter(dataset.get_data_loaders(args.batch_size)[0]))
        imshow_batch(images, labels=labels, normalize=True)

    if args.benchmark:
        if model:
            bm_res = benchmark_inference(
                model,
                dataset,
                args.batch_size,
                args.device,
                num_iterations=args.benchmark_num_iterations,
            )
            if writer:
                plot_confusion_matrix_to_tensorboard(
                    bm_res["labels"],
                    bm_res["predictions"],
                    dataset.get_labels(),
                    writer,
                    args.epochs,
                )
        if model_q:
            q_bm_res = benchmark_inference(
                model_q,
                dataset,
                args.batch_size,
                "cpu",
                num_iterations=args.benchmark_num_iterations,
            )
            if writer:
                plot_confusion_matrix_to_tensorboard(
                    q_bm_res["labels"],
                    q_bm_res["predictions"],
                    dataset.get_labels(),
                    writer,
                    args.epochs,
                )

    export(model, dataset, args)

    if args.tvm_export:
        from integrations import tvm_export

        tvm_export(model, dataset)

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
