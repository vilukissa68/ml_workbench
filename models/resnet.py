import torchvision.models as models
import torch.nn as nn
import torchvision.models.quantization as qmodels


def resnet18(num_classes, args):
    weights = None
    if args.fine_tune:
        if args.weights == "":
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = args.weights

    if args.quantize:
        model = qmodels.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def resnet34(num_classes, args):
    weights = None
    if args.fine_tune:
        if args.weights == "":
            weights = models.ResNet34_Weights.DEFAULT
        else:
            weights = args.weights

    if args.quantize:
        model = qmodels.resnet34(weights=weights)
    else:
        model = models.resnet34(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def resnet50(num_classes, args):
    weights = None
    if args.fine_tune:
        if args.weights == "":
            weights = models.ResNet50_Weights.DEFAULT
        else:
            weights = args.weights

    if args.quantize:
        model = qmodels.resnet50(weights=weights)
    else:
        model = model.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
