#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

def mlperf_tiny_kws(num_classes, args):
    model = DS_CNN()
    model.fc = nn.Linear(model.fc.in_featrues, num_classes)
    return model

class DS_CNN(nn.Module):
    def __init__(self):
        super(DS_CNN, self).__init__()
        self.input_shape = [49, 10, 1]
        self.filters = 64
        self.weight_decay = 1e-4
        self.final_pool_size = (int(self.input_shape[0] / 2), int(self.input_shape[1] / 2))

        # Model layers
        self.conv1 = nn.Conv2d(1, self.filters, kernel_size=(10, 4), stride=(2, 2), padding='same')
        self.bn1 = nn.BatchNorm2d(self.filters)
        self.dropout1 = nn.Dropout(0.2)

        self.dw_conv1 = nn.Conv2d(self.filters, self.filters, kernel_size=(3, 3), padding='same', groups=self.filters)
        self.bn2 = nn.BatchNorm2d(self.filters)
        self.pw_conv1 = nn.Conv2d(self.filters, self.filters, kernel_size=(1, 1), padding='same')
        self.bn3 = nn.BatchNorm2d(self.filters)

        self.dw_conv2 = nn.Conv2d(self.filters, self.filters, kernel_size=(3, 3), padding='same', groups=self.filters)
        self.bn4 = nn.BatchNorm2d(self.filters)
        self.pw_conv2 = nn.Conv2d(self.filters, self.filters, kernel_size=(1, 1), padding='same')
        self.bn5 = nn.BatchNorm2d(self.filters)

        self.dw_conv3 = nn.Conv2d(self.filters, self.filters, kernel_size=(3, 3), padding='same', groups=self.filters)
        self.bn6 = nn.BatchNorm2d(self.filters)
        self.pw_conv3 = nn.Conv2d(self.filters, self.filters, kernel_size=(1, 1), padding='same')
        self.bn7 = nn.BatchNorm2d(self.filters)

        self.dw_conv4 = nn.Conv2d(self.filters, self.filters, kernel_size=(3, 3), padding='same', groups=self.filters)
        self.bn8 = nn.BatchNorm2d(self.filters)
        self.pw_conv4 = nn.Conv2d(self.filters, self.filters, kernel_size=(1, 1), padding='same')
        self.bn9 = nn.BatchNorm2d(self.filters)

        self.dropout2 = nn.Dropout(0.4)
        self.avg_pool = nn.AvgPool2d(kernel_size=self.final_pool_size)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.filters * self.final_pool_size[0] * self.final_pool_size[1], 12)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.dw_conv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pw_conv1(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.dw_conv2(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pw_conv2(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.dw_conv3(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.pw_conv3(x)
        x = self.bn7(x)
        x = F.relu(x)

        x = self.dw_conv4(x)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.pw_conv4(x)
        x = self.bn9(x)
        x = F.relu(x)

        x = self.dropout2(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = F.softmax(x, dim=1)

        return x
