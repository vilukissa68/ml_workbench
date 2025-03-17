from .base_dataset import BaseDataset
import torchvision.transforms as transforms
import os
import numpy as np
import torchaudio
from torchaudio.transforms import MFCC
import torch
from pathlib import Path
import glob
import random
from collections import Counter
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

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

BACKGROUND_FREQUENCY = 0.1
BACKGROUND_VOLUME = 0.1
SAMPLE_RATE = 16000
WINDOW_SIZE_MS = 30.0
WINDOW_STRIDE_MS = 20.0
WINDOW_STRIDE_SAMPLES = int(SAMPLE_RATE * WINDOW_STRIDE_MS / 1000)
WINDOW_SIZE_SAMPLES = int(SAMPLE_RATE * WINDOW_SIZE_MS / 1000)
CLIP_DURATION_MS = 1000
DESIRED_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION_MS / 1000)  # 16000
DCT_COEFFICIENT_COUNT = 10
SILENT_PERCENTAGE = 10.0
LENGTH_MINUS_WINDOW = DESIRED_SAMPLES - WINDOW_SIZE_SAMPLES
SPECTROGRAM_LENGTH = 1 + int(LENGTH_MINUS_WINDOW / WINDOW_STRIDE_SAMPLES)
DATASET_PATH = Path("./data")
BACKGROUND_NOISE_DIR = (
    DATASET_PATH / "SpeechCommands/speech_commands_v0.02/_background_noise_"
)


def silence_waveform(desired_samples, noise_std=0.1):
    noise = torch.randn(desired_samples) * noise_std
    return noise


def get_labels():
    word_labels = [
        "down",
        "go",
        "left",
        "no",
        "off",
        "on",
        "right",
        "stop",
        "up",
        "yes",
        "silence",
        "unknown",
    ]
    return word_labels


def get_idx(label):
    if label in get_labels():
        return get_labels().index(label)
    else:
        return get_labels().index("unknown")


def show_mfcc(mfcc):
    # Plot the MFCC features
    plt.figure(figsize=(10, 4))
    plt.imshow(mfcc, cmap="viridis", origin="lower", aspect="auto")
    plt.title("MFCC Features")
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficient")
    plt.colorbar(label="MFCC Value")
    plt.show()


class MLPerfTinyKWS(BaseDataset):
    def __init__(self, data_path=DATASET_PATH, batch_size=100, shuffle=True):
        super().__init__(data_path=data_path, batch_size=batch_size, shuffle=shuffle)
        self.has_labels = True

    def load_data(self, sets=["train", "test", "val"]):
        if "train" in sets and not self.data_loaded[0]:
            print("Loading training set...")
            train_dataset = SpeechCommandsDataset_pytorch(
                split="training", augment=True
            )
            self.trainset = train_dataset
            print("%d training samples" % len(self.trainset))

        if "test" in sets and not self.data_loaded[1]:
            print("Loading testing set...")
            test_dataset = SpeechCommandsDataset_pytorch(split="testing", augment=False)
            self.testset = test_dataset
            print("%d testing samples" % len(self.testset))

        if "val" in sets and not self.data_loaded[2]:
            print("Loading validation set...")
            val_dataset = SpeechCommandsDataset_pytorch(
                split="validation", augment=False
            )
            self.valset = val_dataset
            print("%d validation samples" % len(self.valset))

        self.data_loaded = (
            self.trainset is not None,
            self.testset is not None,
            self.valset is not None,
        )

    def get_data_shapes(self):
        return {"input_1": (1, 1, 49, 10)}

    def get_num_classes(self):
        return 12

    def get_labels(self):
        return [
            "down",
            "go",
            "left",
            "no",
            "off",
            "on",
            "right",
            "stop",
            "up",
            "yes",
            "silence",
            "unknown",
        ]


class SpeechCommandsDataset_pytorch(Dataset):
    def __init__(self, split="training", augment=True):
        self.split = split
        self.augment = augment
        self.waveforms = []
        self.labels = []
        self.load_data_pytorch()
        print(set(self.labels))
        print(Counter(self.labels))
        if self.augment:
            self.background_data = self.prepare_background_data_pytorch()

    def load_data_pytorch(self):
        dataset = torchaudio.datasets.SPEECHCOMMANDS(
            DATASET_PATH, download=True, subset=self.split
        )

        for waveform, sample_rate, label, *_ in dataset:
            waveform = waveform / waveform.abs().max()  # Normalize
            waveform = waveform.squeeze(0)

            if len(waveform) < DESIRED_SAMPLES:
                waveform = torch.nn.functional.pad(
                    waveform, (0, DESIRED_SAMPLES - len(waveform))
                )

            waveform = torch.nn.functional.pad(
                waveform,
                (2, 2),
                mode="constant",
            )

            # Slice
            waveform = waveform[2 : 2 + DESIRED_SAMPLES]

            if label in get_labels():
                self.waveforms.append(waveform)
                self.labels.append(get_idx(label))
            else:
                pass
                # Add to unknown, keep the same distribution in testing
                if self.split == "testing":
                    if random.random() < 1 / 13:
                        self.waveforms.append(waveform)
                        self.labels.append(int(get_idx("unknown")))
                else:  # Add to unknown in training
                    self.waveforms.append(waveform)
                    self.labels.append(int(get_idx("unknown")))

        silence_count = 0
        if self.split == "training":
            silence_count = int(668)
        else:
            silence_count = int(408)

        for _ in range(silence_count):
            self.waveforms.append(silence_waveform(DESIRED_SAMPLES))
            self.labels.append(get_idx("silence"))

    def prepare_background_data_pytorch(self):
        background_data = []
        if not os.path.exists(BACKGROUND_NOISE_DIR):
            print("No background noise found in " + BACKGROUND_NOISE_DIR)
            return background_data

        search_path = os.path.join(BACKGROUND_NOISE_DIR, "*.wav")
        for wav_path in glob.glob(search_path):
            # Load the audio file
            waveform, sample_rate = torchaudio.load(wav_path, normalize=False)
            # Append the raw audio data to the background_data list
            background_data.append((waveform.squeeze().numpy(), sample_rate))

        if not background_data:
            raise Exception("No background wav files were found in " + search_path)
        return background_data

    def add_background_noise_pytorch(self, waveform):
        if not self.background_data:
            print("No background noise found.")
            return waveform

        noise, sr = random.choice(self.background_data)
        noise = torch.tensor(noise, dtype=torch.float32)
        noise = noise / noise.abs().max()  # Normalize
        if sr != SAMPLE_RATE:
            print("Resampling background noise")
            noise = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(noise)

        if len(noise) < len(waveform):
            noise = torch.nn.functional.pad(noise, (0, len(waveform) - len(noise)))
        if len(noise) > len(waveform):
            noise = noise[: len(waveform)]

        background_volume = 0.0
        if np.random.uniform(0, 1) < BACKGROUND_FREQUENCY:
            background_volume = np.random.uniform(0, BACKGROUND_VOLUME)
        noise = noise * background_volume
        noise_added = waveform + noise
        noise_added = torch.clamp(noise_added, -1.0, 1.0)  # Clamp
        return noise_added

    def get_preprocess_audio_func_pytorch(self, waveform):
        if self.augment:
            waveform = self.add_background_noise_pytorch(waveform)

        # MFCC
        mfcc_transform = MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=DCT_COEFFICIENT_COUNT,
            dct_type=2,
            log_mels=True,
            melkwargs={
                "n_fft": WINDOW_SIZE_SAMPLES,
                "hop_length": WINDOW_STRIDE_SAMPLES,
                "n_mels": 40,
                "f_min": 20.0,
                "f_max": 4000.0,
                "center": False,
            },
        )
        mfcc = mfcc_transform(waveform)
        mfcc = torch.transpose(mfcc, 0, 1)

        # Normalize
        mfcc = (mfcc - mfcc.mean()) / mfcc.std()
        mfcc = mfcc.unsqueeze(0)
        return mfcc

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        waveform = self.waveforms[idx]
        label = self.labels[idx]
        mfcc_features = self.get_preprocess_audio_func_pytorch(waveform)
        return mfcc_features, label
