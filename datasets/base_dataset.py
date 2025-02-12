class BaseDataset:
    def __init__(self, data_path="./data", transform=None):
        self.data_path = data_path
        self.transform = transform
        self.has_labels = None

    def load_data(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_data_loaders(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_data_shapes(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_lables(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_num_classes(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_example_input(self, num=1):
        raise NotImplementedError("Subclasses should implement this method.")
