#!/usr/bin/env python3

from train import train, load_checkpoint, get_data_loaders, get_optimizer
import models
import args_parser
import torchvision.transforms as transforms
from benchmark_inference import benchmark_inference  # Import the benchmarking function
from visualize import imshow_batch
import torch
from quantize import ptsq
from qat import train_qat
from prune import prune_model_global



def load_model(model_name, num_classes, args):
    # Dynamically load the model based on the argument
    model_func = getattr(models, model_name, None)  # Find model function by name
    if model_func is None:
        raise ValueError(f"Model '{model_name}' is not defined in models/resnet.py")

    # Load the model using the selected function
    model = model_func(num_classes=num_classes, args=args)
    return model


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

    train_loader, test_loader, num_classes = get_data_loaders(
        args.dataset, args.batch_size, transform
    )

    if args.load_checkpoint_path:
        model_name = torch.load(args.load_checkpoint_path)["model_name"]
        model = load_model(model_name, num_classes, args)
        optimizer = get_optimizer(args.optimizer, model, args.lr)
        model, optimizer, epoch, quantized, pruned = load_checkpoint(
            model, optimizer, args.load_checkpoint_path
        )
        if quantized:
            model_q = model
            model_q.to("cpu")
    else:  # Load empty model
        model = load_model(args.model, num_classes, args)
        model_q = None

    if args.train or args.fine_tune or args.qat:
        if args.train:
            print("Training model...")
        elif args.fine_tune:
            print("Fine tuning model...")

        if args.qat:
            print("Quantization aware training...")
            model = train_qat(model, train_loader, test_loader, args)
            q_model = model
        else:
            train(model, train_loader, test_loader, args)

    example_input = (next(iter(train_loader))[0][0]).unsqueeze(0)

    # Quantizie fp32, skip if using QAT already
    if args.quantize and not args.qat:
        if args.quantization_method == "ptsq":
            model_q = ptsq(model, train_loader, wrap=False)
        else:
            print("Error! Unknown quantization method.")
            return

        print("Quantization finished.")

    if args.benchmark:
        if not args.qat:
            print("Running inference benchmarking...")
            benchmark_inference(
                model,
                test_loader,
                args.device,
                num_iterations=args.benchmark_num_iterations,
            )
        if model_q:
            print("Running quantized benchmark...")
            benchmark_inference(
                model_q,
                test_loader,
                "cpu",
                num_iterations=args.benchmark_num_iterations,
            )

    if args.prune:
        prune_model_global(model)

    if args.visualize:
        images, labels = next(iter(test_loader))
        imshow_batch(images, labels=labels, normalize=True)

    if args.tvm_export:
        from integrations import tvm_import_pytorch_model, tvm_compile
        mod, params = tvm_import_pytorch_model(model, example_input.shape)
        tvm_compile(mod, params, "./tvm_builds")
        print("TVM compilation finished.")


if __name__ == "__main__":
    main()
