#!/usr/bin/env python3

from train import train, get_optimizer
from export import export, load_checkpoint, load_exported_torch_program
import args_parser
from benchmark_inference import benchmark_inference  # Import the benchmarking function
from visualize import (
    imshow_batch,
    plot_confusion_matrix_to_tensorboard,
    plot_confusion_matrix,
)
import torch
import os
from quantize import ptsq
from qat import train_qat
from prune import prune_model_global
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from utils import get_dataset, load_model

torch.multiprocessing.set_sharing_strategy("file_system")

TIMESTAMP = datetime.now().strftime("%y-%m-%d-%H-%M")


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

    if args.load_checkpoint:
        model, model_q, _ = load_checkpoint(args, dataset)
    elif args.load_exported_program:
        model = load_exported_torch_program(args.load_path)
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

    # Quantizie fp32, skip if using QAT already
    if args.quantize and not args.qat:
        if args.quantization_method == "ptsq":
            model_q = ptsq(model, dataset, args, wrap=True)
        else:
            print("Error! Unknown quantization method.")
            return

    if args.prune:
        if model_q:
            model_q = prune_model_global(model_q)
        else:
            model = prune_model_global(model)

    if args.visualize:
        images, labels = dataset.get_example_input(args.batch_size)
        imshow_batch(images, args, labels=labels, normalize=True)

    if args.benchmark:
        if model:
            bm_res = benchmark_inference(
                model,
                dataset,
                args.device,
                args,
            )
            if writer:
                plot_confusion_matrix_to_tensorboard(
                    bm_res["labels"],
                    bm_res["predictions"],
                    dataset.get_labels(),
                    writer,
                    args.epochs,
                    args,
                )
                plot_confusion_matrix(
                    bm_res["labels"],
                    bm_res["predictions"],
                    dataset.get_labels(),
                    args,
                    title="Confusion Matrix",
                )
        if model_q:
            q_bm_res = benchmark_inference(
                model_q,
                dataset,
                "cpu",
                args,
            )
            if writer:
                plot_confusion_matrix_to_tensorboard(
                    q_bm_res["labels"],
                    q_bm_res["predictions"],
                    dataset.get_labels(),
                    writer,
                    args.epochs,
                    args,
                )
            plot_confusion_matrix(
                q_bm_res["labels"],
                q_bm_res["predictions"],
                dataset.get_labels(),
                args,
                title="Confusion matrix quantized",
            )

    if model:
        export(model, dataset, args)
    if args.save_quantized and model_q:
        export(model_q, dataset, args, quantized=True)

    if args.tvm_export:
        from integrations import tvm_export

        tvm_export(model, dataset)

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
