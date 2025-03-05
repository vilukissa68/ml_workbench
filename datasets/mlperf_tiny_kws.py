from torch.utils.data import Dataset, DataLoader
from .base_dataset import BaseDataset
from .utils import download_file, extract_tar_gz
import torchvision.transforms as transforms

URL = "https://codeload.github.com/eembc/energyrunner/tar.gz/main"
DOWNLOAD_PATH = "./data/kws.tar.gz"
EXTRACT_PATH = "./data/mlperf_tiny_kws"
DATA_PATH = EXTRACT_PATH + "/" + "energyrunner-main/datasets/kws01"

transform = transforms.Compose(
    [
        transforms.Resize((49, 10)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


class MLPerfTinyKWS(BaseDataset):
    def __init__(self, data_path=DATA_PATH, batch_size=64, shuffle=True):
        super().__init__(data_path=data_path, batch_size=batch_size, shuffle=shuffle)
        self.has_labels = True
        self.load_data()

    def load_data(self):
        download_file(URL, DOWNLOAD_PATH)
        extract_tar_gz(DOWNLOAD_PATH, EXTRACT_PATH)
        # untar
        self.trainset = None
        self.testset = None

    def get_data_shapes(self):
        return {"input_1": (1, 49, 10, 1)}

    def get_labels(self):
        return []

    def get_num_classes(self):
        return 12

    def get_example_input(self, num=1):
        example = self.trainset[0][0]
        print(example)
        return example


class KWSDataset(Dataset):
    def __init__(self, csv_file, audio_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            audio_dir (string): Directory with all the audio files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_path = f"{self.audio_dir}/{self.annotations.iloc[idx, 0]}.wav"
        waveform, sample_rate = torchaudio.load(audio_path)
        label = self.annotations.iloc[
            idx, 1
        ]  # Assuming the label is in the second column

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label
