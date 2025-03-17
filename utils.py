#!/usr/bin/env python3

import models
from datasets import mnist, cifar10, mlperf_tiny_kws


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
