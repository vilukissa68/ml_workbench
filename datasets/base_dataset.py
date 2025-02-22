class BaseDataset:
    def __init__(self, data_path="./data", transform=None, batch_size=64, shuffle=True):
        self.data_path = data_path
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.has_labels = None
        self.trainset = None
        self.testset = None
        self.train_loader = None
        self.test_loader = None

    def load_data(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_data_loaders(self):
        return self.train_loader, self.test_loader

    def get_data_shapes(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_lables(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_num_classes(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_example_input(self, num=1):
        example, _ = next(iter(self.train_loader))
        return example
