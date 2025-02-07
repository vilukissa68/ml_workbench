import torch
import torch.nn as nn
import torch.quantization as quantization


class SimpleConvModel(nn.Module):
    def __init__(self, output_size=10, quantizable=False):
        super(SimpleConvModel, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, output_size)  # Custom output size

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.softmax = nn.Softmax(dim=1)

        # Quantization related layers
        self.quantizable = quantizable
        if self.quantizable:
            # Add Quantize and DeQuantize layers manually
            self.quant = quantization.QuantStub()  # Quantize input
            self.dequant = quantization.DeQuantStub()  # DeQuantize output

    def forward(self, x):
        if self.quantizable:
            x = self.quant(x)  # Quantize input

        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        if self.quantizable:
            x = self.dequant(x)  # DeQuantize output

        return self.softmax(x)


def simpleconvmodel(num_classes, args):
    model = SimpleConvModel(num_classes, args.quantize)
    return model
