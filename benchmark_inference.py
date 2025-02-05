#!/usr/bin/env python3
import time
import torch


def benchmark_inference(model, data_loader, device, num_iterations):
    model.eval()
    model.to(device)

    # Measure time for the first few iterations to warm-up the model
    for _ in range(5):
        inputs, _ = next(iter(data_loader))
        inputs = inputs.to(device)
        with torch.no_grad():
            _ = model(inputs)

    # Start benchmarking
    total_time = 0
    for _ in range(num_iterations):
        inputs, _ = next(iter(data_loader))
        inputs = inputs.to(device)

        # Record time for inference
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs)
        end_time = time.time()

        total_time += end_time - start_time

    avg_latency = total_time / num_iterations
    print(f"Average Inference Latency: {avg_latency * 1000:.4f} ms per batch")
