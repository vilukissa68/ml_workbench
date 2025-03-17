import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from export import save_checkpoint
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from optimizers import get_optimizer


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def is_main_process():
    if not is_dist_avail_and_initialized():
        return True
    rank = dist.get_rank()
    return rank == 0


def get_criterion(criterion):
    if criterion == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    elif criterion == "MSELoss":
        return nn.MSELoss()
    elif criterion == "L1Loss":
        return nn.L1Loss()
    elif criterion == "NLLLoss":
        return nn.NLLLoss()
    elif criterion == "BCELoss":
        return nn.BCELoss()
    elif criterion == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()
    else:
        raise Exception("Error:{criterion} is not a valid critetion!".format(criterion))


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
    regularization="",
    lambda_reg=0.01,
    verbose=False,
):
    model.train(True)
    running_loss = 0.0
    correct = 0
    total = 0
    torch.cuda.empty_cache()

    for inputs, labels in tqdm(data_loader, desc="Training", disable=not verbose):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # Apply L1 regularization
        if regularization == "L1":
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss += lambda_reg * l1_norm

        # Apply L2 regularization
        elif regularization == "L2":
            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
            loss += lambda_reg * l2_norm

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total

    return epoch_loss, accuracy


def evaluate(model, data_loader, criterion, device, verbose=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating", disable=not verbose):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total

    return epoch_loss, accuracy


def train(
    gpu,
    model,
    dataset,
    args,
    writer=None,
):

    # Set seed
    torch.manual_seed(args.seed)
    device = args.device
    num_epochs = args.epochs
    lr = args.lr
    optimizer_type = args.optimizer
    verbose = args.verbose
    criterion = get_criterion(args.criterion)
    optimizer, scheduler = get_optimizer(model, args)

    # Load data to model
    dataset.load_data()

    if args.verbose:
        print("Training on", device)

    if args.distributed_training:
        rank = args.local_rank * args.ngpus + gpu
        model.cuda(gpu)
        torch.cuda.set_device(gpu)
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=args.world_size, rank=rank
        )
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu], find_unused_parameters=False
        )
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset.trainset, num_replicas=args.world_size, rank=rank
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset.trainset, num_replicas=args.world_size, rank=rank
        )
        train_loader, test_loader, _ = dataset.get_data_loaders(
            args.batch_size, train_sampler, None, 0
        )
        print("Rank:", dist.get_rank())
    else:
        train_loader, test_loader, _ = dataset.get_data_loaders(
            args.batch_size, None, None, args.num_workers
        )
        model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Train the model
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.regularization,
            args.reg_lambda,
            verbose,
        )
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Acc/train", train_acc, epoch)
            writer.flush()

        # Evaluation
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, verbose)

        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

        if writer:
            writer.add_scalar("Loss/test", test_loss, epoch)
            writer.add_scalar("Acc/test", test_acc, epoch)
            writer.flush()

        # Save checkpoint after each epoch overwriting the previous epochs checkpoint
        if args.distributed_training:
            dist.barrier()
        if is_main_process():
            save_checkpoint(
                model,
                args,
                optimizer=optimizer,
                optimizer_name=optimizer_type,
                epoch=epoch,
                learning_rate=lr,
                model_name=args.model,
                dataset_name=args.dataset,
                batch_size=args.batch_size,
                checkpoint_dir=args.checkpoint_dir,
            )

        # Update the learning rate
        if scheduler:
            scheduler.step()
            if args.verbose:
                print(f"New scheduled learning rate: {scheduler.get_last_lr()}")

    if verbose:
        print("Training Complete!")

    if writer:
        images, _ = next(iter(train_loader))
        images.to("cpu")
        model.to("cpu")
        writer.add_graph(model, images)
        writer.flush()
        model.to(device)

    if args.distributed_training:
        dist.destroy_process_group()


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    quantized = checkpoint["quantized"]
    pruned = checkpoint["pruned"]
    return model, optimizer, epoch, quantized, pruned
