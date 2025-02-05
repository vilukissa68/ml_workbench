#!/usr/bin/env python3

import train
from models import resnet
import args_parser
import torchvision.transforms as transforms
from benchmark_inference import benchmark_inference  # Import the benchmarking function


def load_model(model_name, num_classes=10):
    # Dynamically load the model based on the argument
    model_func = getattr(resnet, model_name, None)  # Find model function by name
    if model_func is None:
        raise ValueError(f"Model '{model_name}' is not defined in models/resnet.py")

    # Load the model using the selected function
    model = model_func(num_classes=num_classes)
    return model


def main():
    args = args_parser.parse_args()

    print(f"Arguments: {args}")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_loader, test_loader = train.get_data_loaders(
        args.dataset, args.batch_size, transform
    )

    model = load_model(args.model, 10)

    if args.train:
        print("Training the model...")
        train.train(model, train_loader, test_loader, args)

    if args.quantize:
        # TODO: Add quantization logic
        pass

    if args.benchmark:
        print("Running inference benchmarking...")
        benchmark_inference(
            model, test_loader, device, num_iterations=args.num_iterations
        )


if __name__ == "__main__":
    main()
