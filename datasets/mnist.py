import torchvision
from .base_dataset import BaseDataset
from torch.utils.data import DataLoader


class MNIST(BaseDataset):
    def __init__(self):
        super(BaseDataset, self).__init__()
        self.has_labels = False

    def load_data(self):
        self.trainset = trainset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        self.testset = testset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

    def get_data_loaders(self, batch_size=64, shuffle=True):
        train_loader = DataLoader(self.trainset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(self.testset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    def get_data_shapes(self):
        return {"input0": (1, 28, 28, 1)}

    def get_num_classes(self):
        return 10

    def get_example_input(self, num=1):
        return self.trainset[0:num]
