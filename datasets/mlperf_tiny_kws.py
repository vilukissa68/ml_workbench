from torch.utils.data import Dataset, DataLoader
from .base_dataset import BaseDataset
from .utils import download_file, extract_tar_gz
import torchvision.transforms as transforms
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.platform import gfile
import numpy as np
import torchaudio
from torchaudio.transforms import MFCC
import torch
from pathlib import Path
import glob
import random
import wavfile

transform = transforms.Compose(
    [
        transforms.Resize((49, 10)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

NUM_TRAIN_SAMPLES = -1
NUM_VAL_SAMPLES = -1
NUM_TEST_SAMPLES = -1
NUM_MFCCS = 10

BACKGROUND_FREQUENCY = 0.8
BACKGROUND_VOLUME = 0.1
SAMPLE_RATE = 16000
WINDOW_SIZE_MS = 30.0
WINDOW_STRIDE_MS = 20.0
WINDOW_STRIDE_SAMPLES = int(SAMPLE_RATE * WINDOW_STRIDE_MS / 1000)
WINDOW_SIZE_SAMPLES = int(SAMPLE_RATE * WINDOW_SIZE_MS / 1000)
CLIP_DURATION_MS = 1000
DESIRED_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION_MS / 1000)  # 16000
DCT_COEFFICIENT_COUNT = 10
LENGTH_MINUS_WINDOW = DESIRED_SAMPLES - WINDOW_SIZE_SAMPLES
SPECTROGRAM_LENGTH = 1 + int(LENGTH_MINUS_WINDOW / WINDOW_STRIDE_SAMPLES)
DATASET_PATH = Path("./data")
BACKGROUND_NOISE_DIR = (
    DATASET_PATH / "SpeechCommands/speech_commands_v0.02/_background_noise_"
)


class MLPerfTinyKWS(BaseDataset):
    def __init__(self, data_path=DATASET_PATH, batch_size=64, shuffle=True):
        super().__init__(data_path=data_path, batch_size=batch_size, shuffle=shuffle)
        self.has_labels = True
        self.load_data()

    def load_data(self):
        print("Loading training data...")
        # train_dataset = SpeechCommandsDataset_pytorch(split="training", augment=True)

        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        print("Loading test data...")
        test_dataset = SpeechCommandsDataset_pytorch(split="testing", augment=True)

        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True
        )

        # self.trainset = train_dataset
        self.testset = test_dataset
        print("Data loaded.")
        # print("Train set size:", len(self.trainset))
        print("Test set size:", len(self.testset))
        print("Input:", next(iter(test_loader))[0])

    def get_data_shapes(self):
        return {"input_1": (1, 49, 10, 1)}

    def get_labels(self):
        word_labels = [
            "Down",
            "Go",
            "Left",
            "No",
            "Off",
            "On",
            "Right",
            "Stop",
            "Up",
            "Yes",
            "Silence",
            "Unknown",
        ]
        return word_labels

    def get_num_classes(self):
        return 12

    def get_example_input(self, num=1):
        example = self.trainset[0][0]
        print(example)
        return example


class SpeechCommandsDataset_pytorch(Dataset):
    def __init__(self, split="train", augment=True):
        self.split = split
        self.augment = augment
        self.waveforms = []
        self.labels = []
        self.cache = {}
        self.use_cache = False
        self.load_data_pytorch()
        if self.augment:
            self.background_data = self.prepare_background_data_pytorch()

    def load_data_pytorch(self):
        dataset = torchaudio.datasets.SPEECHCOMMANDS(
            DATASET_PATH, download=True, subset=self.split
        )
        print("Dataset length:", len(dataset))

        for waveform, sample_rate, label, *_ in dataset:
            if sample_rate != SAMPLE_RATE:
                print("Resampling...")
                waveform = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)(
                    waveform
                )
            self.waveforms.append(waveform)
            self.labels.append(label)

    def prepare_background_data_pytorch(self):
        print("Preparing background data...")
        background_data = []
        if not os.path.exists(BACKGROUND_NOISE_DIR):
            print("No background noise found in " + BACKGROUND_NOISE_DIR)
            return background_data

        search_path = os.path.join(BACKGROUND_NOISE_DIR, "*.wav")
        print("Extracting background data...")
        for wav_path in glob.glob(search_path):
            # Load the audio file
            waveform, sample_rate = torchaudio.load(wav_path)
            # Append the raw audio data to the background_data list
            background_data.append((waveform.squeeze().numpy(), sample_rate))

        if not background_data:
            raise Exception("No background wav files were found in " + search_path)
        return background_data

    def add_background_noise_pytorch(self, waveform):
        if not self.background_data:
            print("No background noise found.")
            return waveform

        print("Adding background noise...")
        noise, sr = random.choice(self.background_data)
        print("Noise shape:", noise.shape)
        noise = torch.tensor(noise, dtype=torch.float32) / 32768.0
        if sr != SAMPLE_RATE:
            noise = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(noise)

        print("Noise shape:", noise.shape)
        noise = noise[: waveform.shape[0]]  # Trim or pad noise
        noise *= random.uniform(0, 0.1)  # Scale noise
        return waveform + noise

    def get_preprocess_audio_func_pytorch(self, waveform):
        print("Preprocessing audio...")
        waveform = waveform.squeeze(0)
        print("Waveform shape:", waveform.shape)
        waveform.float()  # Ensure int16
        waveform = waveform / waveform.abs().max()  # Normalize

        # Pad to desired length
        if waveform.shape[-1] < DESIRED_SAMPLES:
            waveform = torch.nn.functional.pad(
                waveform, (0, DESIRED_SAMPLES - waveform.shape[-1])
            )
        elif waveform.shape[-1] > DESIRED_SAMPLES:
            waveform = waveform[:DESIRED_SAMPLES]
        print("Padded waveform shape:", waveform.shape)

        time_shift_padding_placeholder_ = torch.tensor([2, 2], dtype=torch.int32)
        time_shift_offset_placeholder_ = torch.tensor([2], dtype=torch.int32)
        waveform = torch.nn.functional.pad(
            waveform,
            (
                time_shift_padding_placeholder_[0].item(),
                time_shift_padding_placeholder_[1].item(),
            ),
        )
        waveform = waveform[
            time_shift_offset_placeholder_[0]
            .item() : time_shift_offset_placeholder_[0]
            .item()
            + DESIRED_SAMPLES
        ]
        print("Padded waveform shape:", waveform.shape)

        if self.augment:
            waveform = self.add_background_noise_pytorch(waveform)

        # MFCC
        mfcc_transform = MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=DCT_COEFFICIENT_COUNT,
            log_mels=True,
            melkwargs={
                "n_fft": WINDOW_SIZE_SAMPLES,
                "hop_length": WINDOW_STRIDE_SAMPLES,
                "n_mels": 40,
                "f_min": 20,
                "f_max": 4000,
                "center": False,
            },
        )
        mfcc = mfcc_transform(waveform)
        print("MFCC shape:", mfcc.shape)
        return torch.reshape(mfcc, (1, SPECTROGRAM_LENGTH, DCT_COEFFICIENT_COUNT, 1))

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        waveform = self.waveforms[idx]
        label = self.labels[idx]

        # Check if MFCC features are cached
        if str(idx) in self.cache and self.use_cache:
            mfcc_features = self.cache[str(idx)]
        else:
            # Cache MFCC features
            mfcc_features = self.get_preprocess_audio_func_pytorch(waveform)
            if self.use_cache:
                self.cache[str(idx)] = mfcc_features
        return mfcc_features, label
