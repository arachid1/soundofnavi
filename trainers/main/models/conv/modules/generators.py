from .helpers import *

import numpy as np

import nlpaug.flow as naf
import nlpaug.augmenter.spectrogram as nas

class data_generator(tf.keras.utils.Sequence):
    def __init__(
        self, train_data, sr, n_classes, shape, batch_size, initial_channels, shuffle=True,
    ):
        self.train_data = train_data
        self.n_classes = n_classes
        self.shape = shape
        self.batch_size = batch_size
        self.initial_channels = initial_channels
        self.timesteps = 0
        self.shuffle = True
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.train_data) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index *
                               self.batch_size: (index + 1) * self.batch_size]
        wav_batch = [self.train_data[k] for k in indexes]

        # returns labels and the longest timestamp
        wav_labels = label_data(wav_batch)

        final_shape = (self.batch_size,
                       self.shape[0], self.shape[1], self.initial_channels)

        X = np.zeros(final_shape, dtype=np.float32)
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (wav, label) in enumerate(zip(wav_batch, wav_labels)):
            wav = wav[0]
            wav = np.repeat(wav[..., np.newaxis], self.initial_channels, -1)
            X[i, :, :] = wav
            Y[i, ] = multi_hot_encoding(label)
        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.train_data))
        if self.shuffle:
            np.random.shuffle(self.indexes)