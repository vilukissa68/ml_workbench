# Model paper https://arxiv.org/abs/2201.03545
import torchvision.models as models
import torch.nn as nn
import torchvision.models.quantization as qmodels


def convnext_tiny(num_classes, args):
    weights = None
    if args.fine_tune:
        if args.weights == "":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        else:
            weights = args.weights
    if args.quantize:
        raise Exception("Error! ConvNext doesn't currently support quantization!")
    else:
        model = models.convnext_tiny(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model


def convnext_small(num_classes, args):
    weights = None
    if args.fine_tune:
        if args.weights == "":
            weights = models.ConvNeXt_Small_Weights.DEFAULT
        else:
            weights = args.weights
    if args.quantize:
        raise Exception("Error! ConvNext doesn't currently support quantization!")
    else:
        model = models.convnext_small(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model


def convnext_base(num_classes, args):
    weights = None
    if args.fine_tune:
        if args.weights == "":
            weights = models.ConvNeXt_Base_Weights.DEFAULT
        else:
            weights = args.weights
    if args.quantize:
        raise Exception("Error! ConvNext doesn't currently support quantization!")
    else:
        model = models.convnext_base(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model


def convnext_large(num_classes, args):
    weights = None
    if args.fine_tune:
        if args.weights == "":
            weights = models.ConvNeXt_Large_Weights.DEFAULT
        else:
            weights = args.weights
    if args.quantize:
        raise Exception("Error! ConvNext doesn't currently support quantization!")
    else:
        model = models.convnext_large(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model
