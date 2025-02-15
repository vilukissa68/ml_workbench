import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os
from datasets import mnist, cifar10
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def get_optimizer(optimizer_type, model, lr):
    if optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Optimizer {optimizer_type} is not supported!")
    return optimizer


def train_one_epoch(model, data_loader, criterion, optimizer, device, verbose=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(data_loader, desc="Training", disable=not verbose):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
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
    model,
    dataset,
    args,
    writer=None,
):
    device = args.device
    num_epochs = args.epochs
    lr = args.lr
    optimizer_type = args.optimizer
    verbose = args.verbose
    batch_size = args.batch_size
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args.optimizer, model, lr)

    train_loader, test_loader = dataset.get_data_loaders(args.batch_size)

    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Train the model
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, verbose
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
        save_checkpoint(
            model,
            optimizer,
            epoch,
            model_name=args.model,
            dataset_name=args.dataset,
            batch_size=batch_size,
            checkpoint_dir=args.checkpoint_dir,
            quantized=False,
            pruned=False,
        )

    print("Training Complete!")

    if writer:
        images, _ = next(iter(train_loader))
        images.to("cpu")
        model.to("cpu")
        print(images)
        writer.add_graph(model, images)
        writer.flush()
        writer.close()
        model.to(device)


def save_checkpoint(
    model,
    optimizer,
    epoch,
    model_name,
    dataset_name,
    batch_size,
    checkpoint_dir,
    quantized=False,
    pruned=False,
):
    # Create the checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if quantized:
        quantization_info = "q"
    else:
        quantization_info = "no_q"

    if pruned:
        pruning_info = "pruned"
    else:
        pruning_info = "no_prune"

    checkpoint_filename = f"{model_name}_{dataset_name}_{batch_size}_{epoch+1}_{quantization_info}_{pruning_info}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    torch.save(
        {
            "model_name": model_name,
            "epoch": epoch,
            "quantized": quantized,
            "pruned": pruned,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )

    print(f"Checkpoint saved at {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    quantized = checkpoint["quantized"]
    pruned = checkpoint["pruned"]
    return model, optimizer, epoch, quantized, pruned
