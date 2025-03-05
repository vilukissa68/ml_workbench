import torchvision
from .base_dataset import BaseDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


class MNIST(BaseDataset):
    def __init__(self, batch_size=64, shuffle=True):
        super(BaseDataset, self).__init__()
        self.has_labels = False
        load_data()

    def load_data(self):
        self.trainset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        self.testset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

    def get_data_shapes(self):
        return {"input0": (1, 28, 28, 1)}

    def get_num_classes(self):
        return 10

    def get_example_input(self, num=1):
        return self.trainset[0:num]

    def get_labels(self):
        return [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
        ]
