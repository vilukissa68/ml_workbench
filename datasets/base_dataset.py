from torch.utils.data import DataLoader


class BaseDataset:
    def __init__(self, data_path="./data", transform=None, batch_size=64, shuffle=True):
        self.data_path = data_path
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.has_labels = None
        self.trainset = None
        self.testset = None

    def load_data(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_data_loaders(
        self, batch_size=64, train_sampler=None, test_sampler=None, num_workers=0
    ):
        if train_sampler:
            shuffle = False
        else:
            shuffle = self.shuffle
        train_loader = DataLoader(
            self.trainset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            self.testset,
            batch_size=batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        return train_loader, test_loader

    def get_data_shapes(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_lables(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_num_classes(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_example_input(self, num=1):
        train_loader, _ = self.get_data_loaders(num, False)
        example, _ = next(iter(train_loader))
        return example
