import torchvision
from .base_dataset import BaseDataset
from torch.utils.data import DataLoader


class CIFAR10(BaseDataset):
    def __init__(self, data_path="./data", transform=None):
        super().__init__(data_path=data_path, transform=transform)
        self.has_labels = True
        self.transform = transform
        self.load_data()

    def load_data(self):
        self.trainset = torchvision.datasets.CIFAR10(
            root=self.data_path, train=True, download=True, transform=self.transform
        )
        self.testset = torchvision.datasets.CIFAR10(
            root=self.data_path, train=False, download=True, transform=self.transform
        )

    def get_data_loaders(self, batch_size=64, shuffle=True):
        train_loader = DataLoader(self.trainset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(self.testset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    def get_data_shapes(self):
        return {"input0": (1, 32, 32, 3)}

    def get_labels(self):
        return [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

    def get_num_classes(self):
        return 10

    def get_example_input(self, num=1):
        example = self.trainset[0][0]
        return example
