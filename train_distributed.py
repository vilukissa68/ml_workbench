#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def train_data_parallel(model, dataset, rank, world_size, args):
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}")
    num_epochs = args.epochs
    lr = args.lr
    optimizer_type = args.optimizer
    verbose = args.verbose

    # Model
    model = args.model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args.optimizer, model, lr)

    # Load data with DistributedSampler
    train_dataset, test_dataset = dataset.get_datasets()
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Training loop
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)  # Shuffle dataset for DDP
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, verbose)

        if rank == 0:
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

        # Evaluation (only rank 0 prints)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, verbose)

        if rank == 0:
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

        # Save checkpoint (only rank 0)
        if rank == 0:
            save_checkpoint(model, args, optimizer, optimizer_type, epoch, lr, args.model, args.dataset, args.batch_size, args.checkpoint_dir)

    if rank == 0:
        print("Training Complete!")

    # Cleanup
    dist.destroy_process_group()
