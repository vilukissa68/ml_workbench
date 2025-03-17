import time
import torch
import torch.nn.functional as F


def benchmark_inference(model, dataset, device, args):
    """
    Benchmarks the inference latency and measures accuracy of the model.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataset (Dataset): The dataset to evaluate the model on.
        device (str): The device to run the inference on.
        args (argparse.Namespace): The command-line arguments.

    Returns:
        dict: A dictionary containing the accuracy, average latency, and predictions
    """

    model.eval()
    model.to(device)

    if args.verbose:
        # Print detailed inference specs
        print(
            "Inference benchmarking on {}, batch size {}...".format(
                device, args.batch_size
            )
        )

    correct = 0
    total = 0

    dataset.load_data(["test"])
    _, data_loader, _ = dataset.get_data_loaders()

    # Warm-up
    for _ in range(5):
        inputs, _ = next(iter(data_loader))
        inputs = inputs.to(device)
        with torch.no_grad():
            _ = model(inputs)

    # Start benchmarking
    total_time = 0
    all_preds = []
    all_lables = []
    for i, (inputs, labels) in enumerate(data_loader):
        if i >= args.benchmark_num_iterations:
            break

        inputs, labels = inputs.to(device), labels.to(device)

        # Record time for inference
        start_time = time.time()
        with torch.no_grad():
            outputs = model(inputs)
        end_time = time.time()

        total_time += end_time - start_time

        # Compute accuracy
        preds = outputs.argmax(dim=1)  # Get the predicted class
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.to("cpu"))
        all_lables.extend(labels.to("cpu"))

    avg_latency = total_time / args.benchmark_num_iterations
    accuracy = correct / total * 100

    print(
        f"Average Inference Latency: {avg_latency * 1000:.4f} ms per batch (batch size = {args.batch_size})"
    )
    print(f"Accuracy: {accuracy:.2f}%")

    return {
        "accuracy": accuracy,
        "avg_latency": avg_latency * 1000,
        "predictions": all_preds,
        "labels": all_lables,
    }  # Latency in ms
