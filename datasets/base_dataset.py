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
        self.valset = None
        self.labels = []
        self.data_loaded = (False, False, False)

    def load_data(self, sets=["train", "test", "val"]):
        self.data_loaded = (True, True, True)

    def get_data_loaders(
        self,
        batch_size=64,
        train_sampler=None,
        test_sampler=None,
        num_workers=0,
        force_shuffle=False,
    ):

        # Default shuffles
        train_shuffle = self.shuffle
        test_shuffle = False
        val_shuffle = False

        if train_sampler:
            train_shuffle = False

        if force_shuffle:
            train_shuffle = True
            test_shuffle = True
            val_shuffle = True

        train_loader = None
        test_loader = None
        val_loader = None

        if self.trainset:
            train_loader = DataLoader(
                self.trainset,
                batch_size=batch_size,
                sampler=train_sampler,
                shuffle=train_shuffle,
                num_workers=num_workers,
                pin_memory=True,
            )
        if self.testset:
            test_loader = DataLoader(
                self.testset,
                batch_size=batch_size,
                shuffle=test_shuffle,
                sampler=test_sampler,
                num_workers=num_workers,
                pin_memory=True,
            )
        if self.valset:
            val_loader = DataLoader(
                self.valset,
                batch_size=batch_size,
                shuffle=val_shuffle,
                num_workers=num_workers,
                pin_memory=True,
            )
        return train_loader, test_loader, val_loader

    def get_data_shapes(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_labels(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_num_classes(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_example_input(self, num=1, shuffle=True):
        # Check if any dataset is loaded already
        if self.data_loaded[0]:
            data_loader, _, _ = self.get_data_loaders(num, force_shuffle=shuffle)
        elif self.data_loaded[1]:
            _, data_loader, _ = self.get_data_loaders(num, force_shuffle=shuffle)
        elif self.data_loaded[2]:
            _, _, data_loader = self.get_data_loaders(num, force_shuffle=shuffle)
        else:
            self.load_data(["test"])
            _, data_loader, _ = self.get_data_loaders(num, force_shuffle=shuffle)

        example, label = next(iter(data_loader))
        return (
            example,
            label,
        )
