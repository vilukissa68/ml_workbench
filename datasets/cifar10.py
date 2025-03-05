import torchvision
from .base_dataset import BaseDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


class CIFAR10(BaseDataset):
    def __init__(
        self, data_path="./data", transform=transform
    ):
        super().__init__(
            data_path=data_path,
            transform=transform,
        )
        self.has_labels = True
        self.load_data()

    def load_data(self):
        self.trainset = torchvision.datasets.CIFAR10(
            root=self.data_path, train=True, download=True, transform=self.transform
        )
        self.testset = torchvision.datasets.CIFAR10(
            root=self.data_path, train=False, download=True, transform=self.transform
        )

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
