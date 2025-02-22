import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
from train import save_checkpoint, get_optimizer
from tqdm import tqdm


def fuse_layers(model):
    """Automatically fuses Conv + BN + ReLU layers where applicable."""
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            layers = list(module.named_children())
            fuse_list = []
            for i in range(len(layers) - 1):
                layer_name, layer = layers[i]
                next_layer_name, next_layer = layers[i + 1]
                if isinstance(layer, nn.Conv2d) and isinstance(
                    next_layer, nn.BatchNorm2d
                ):
                    if i + 2 < len(layers) and isinstance(layers[i + 2][1], nn.ReLU):
                        fuse_list.append(
                            [layer_name, next_layer_name, layers[i + 2][0]]
                        )
                    else:
                        fuse_list.append([layer_name, next_layer_name])

            if fuse_list:
                torch.quantization.fuse_modules(module, fuse_list, inplace=True)


def evaluate(model, test_loader, device="cpu"):
    """Evaluates model accuracy."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def train_qat(model, dataset, args):
    device = args.device
    num_epochs = args.epochs
    lr = args.lr
    optimizer_type = args.optimizer
    verbose = args.verbose
    batch_size = args.batch_size
    backend = args.quantization_backend

    train_loader, test_loader = data.get_data_loaders()

    model.to(device)
    model.train()

    optimizer = get_optimizer(args.optimizer, model, lr)
    criterion = nn.CrossEntropyLoss()

    # Fuse layers
    fuse_layers(model)

    # Prepare model for QAT
    torch.backends.quantized.engine = backend
    model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    torch.quantization.prepare_qat(model, inplace=True)

    # Train the QAT model
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        train_iter = (
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            if verbose
            else train_loader
        )

        for inputs, labels in train_iter:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
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

    print("QAT Training complete.")

    # Evaluate the QAT model
    evaluate(model, test_loader, device)

    # Convert to a fully quantized model
    model.eval()
    torch.quantization.convert(model, inplace=True)
    print("Model successfully converted to quantized version.")

    # Evaluate quantized model
    evaluate(model, test_loader, device)

    return model
