from torch.utils.data import Dataset, DataLoader
from .base_dataset import BaseDataset
from .utils import download_file, extract_tar_gz
import torchvision.transforms as transforms
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.platform import gfile

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
        train, test, val = get_training_data()
        # untar
        self.trainset = None
        self.testset = None

    def get_data_shapes(self):
        return {"input_1": (1, 49, 10, 1)}

    def get_labels(self):
        word_labels = ["Down", "Go", "Left", "No", "Off", "On", "Right",
               "Stop", "Up", "Yes", "Silence", "Unknown"]
        return word_labels

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

def get_training_data(background_frequency=1000, background_volume=10):
    label_count=12

    # Set path for background noise and load
    bg_path="data/background"
    BACKGROUND_NOISE_DIR_NAME='_background_noise_'
    background_data = prepare_background_data(bg_path,BACKGROUND_NOISE_DIR_NAME)

    # Load waveforms data
    splits = ['train', 'test', 'validation']
    (ds_train, ds_test, ds_val), ds_info = tfds.load('speech_commands', split=splits,
                                                    data_dir="data", with_info=True)

    # Get subsets
    if Flags.num_train_samples != -1:
        ds_train = ds_train.take(Flags.num_train_samples)
    if Flags.num_val_samples != -1:
        ds_val = ds_val.take(Flags.num_val_samples)
    if Flags.num_test_samples != -1:
        ds_test = ds_test.take(Flags.num_test_samples)

    # extract spectral features and add background noise
    ds_train = ds_train.map(get_preprocess_audio_func(model_settings,is_training=True,
                                                    background_data=background_data),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test  =  ds_test.map(get_preprocess_audio_func(model_settings,is_training=False,
                                                    background_data=background_data),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val   =   ds_val.map(get_preprocess_audio_func(model_settings,is_training=False,
                                                    background_data=background_data),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # change output from a dictionary to a feature,label tuple
    ds_train = ds_train.map(convert_dataset)
    ds_test = ds_test.map(convert_dataset)
    ds_val = ds_val.map(convert_dataset)

    # Now that we've acquired the preprocessed data, either by processing or loading,
    ds_train = ds_train.batch(Flags.batch_size)
    ds_test = ds_test.batch(Flags.batch_size)
    ds_val = ds_val.batch(Flags.batch_size)

    return ds_train, ds_test, ds_val


def prepare_background_data(bg_path,BACKGROUND_NOISE_DIR_NAME):
  """Searches a folder for background noise audio, and loads it into memory.
  It's expected that the background audio samples will be in a subdirectory
  named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
  the sample rate of the training data, but can be much longer in duration.
  If the '_background_noise_' folder doesn't exist at all, this isn't an
  error, it's just taken to mean that no background noise augmentation should
  be used. If the folder does exist, but it's empty, that's treated as an
  error.
  Returns:
    List of raw PCM-encoded audio samples of background noise.
  Raises:
    Exception: If files aren't found in the folder.
  """
  background_data = []
  background_dir = os.path.join(bg_path, BACKGROUND_NOISE_DIR_NAME)
  if not os.path.exists(background_dir):
    return background_data
  #with tf.Session(graph=tf.Graph()) as sess:
  #    wav_filename_placeholder = tf.placeholder(tf.string, [])
  #    wav_loader = io_ops.read_file(wav_filename_placeholder)
  #    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
  search_path = os.path.join(bg_path, BACKGROUND_NOISE_DIR_NAME,'*.wav')
  #for wav_path in gfile.Glob(search_path):
  #    wav_data = sess.run(wav_decoder, feed_dict={wav_filename_placeholder: wav_path}).audio.flatten()
  #    self.background_data.append(wav_data)
  for wav_path in gfile.Glob(search_path):
    #audio = tfio.audio.AudioIOTensor(wav_path)
    raw_audio = tf.io.read_file(wav_path)
    audio = tf.audio.decode_wav(raw_audio)
    background_data.append(audio[0])
  if not background_data:
    raise Exception('No background wav files were found in ' + search_path)
  return background_data
