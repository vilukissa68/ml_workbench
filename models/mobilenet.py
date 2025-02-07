import torchvision.models as models
import torch.nn as nn
import torchvision.models.quantization as qmodels


def mobilenetv2(num_classes, args):
    weights = None
    if args.fine_tune:
        if args.weights == "":
            weights = models.MobileNet_V2_Weights.DEFAULT
        else:
            weights = args.weights

    if args.quantize:
        model = qmodels.mobilenet_v2(weights=weights)
    else:
        model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def mobilenetv3(num_classes, args):
    weights = None
    if args.fine_tune:
        if args.weights == "":
            weights = models.MobileNet_V3_Weights.DEFAULT
        else:
            weights = args.weights

    if args.quantize:
        model = qmodels.mobilenet_v3(weights=weights)
    else:
        model = models.mobilenet_v3(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model
