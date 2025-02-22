import time
import torch
import torch.nn.functional as F


def benchmark_inference(model, dataset, batch_size, device, num_iterations):
    """
    Benchmarks the inference latency and measures accuracy of the model.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader for evaluation.
        device (torch.device): Device to run inference on (CPU or CUDA).
        num_iterations (int): Number of iterations for benchmarking.

    Returns:
        dict: Contains 'accuracy' and 'avg_latency' in milliseconds.
    """

    model.eval()
    model.to(device)

    correct = 0
    total = 0

    _, data_loader = dataset.get_data_loaders()

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
        if i >= num_iterations:
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

    avg_latency = total_time / num_iterations
    accuracy = correct / total * 100

    print(
        f"Average Inference Latency: {avg_latency * 1000:.4f} ms per batch (batch size = {batch_size})"
    )
    print(f"Accuracy: {accuracy:.2f}%")

    return {
        "accuracy": accuracy,
        "avg_latency": avg_latency * 1000,
        "predictions": all_preds,
        "labels": all_lables,
    }  # Latency in ms
