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
from export import save_checkpoint


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
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args.optimizer, model, lr)

    train_loader, test_loader = dataset.get_data_loaders()

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

    print("Training Complete!")

    if writer:
        images, _ = next(iter(train_loader))
        images.to("cpu")
        model.to("cpu")
        writer.add_graph(model, images)
        writer.flush()
        model.to(device)


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    quantized = checkpoint["quantized"]
    pruned = checkpoint["pruned"]
    return model, optimizer, epoch, quantized, pruned
