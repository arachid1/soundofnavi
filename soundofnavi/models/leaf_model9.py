import pandas as pd
import numpy as np
import tensorflow as tf
from .core import *
# from tensorflow.keras import layers
from keras.layers import ReLU, BatchNormalization, Concatenate
from keras.layers.core import Dense, Flatten, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D

from tensorflow.keras.regularizers import l2
from kapre.composed import get_melspectrogram_layer
from kapre.time_frequency import STFT, Magnitude, Phase, MagnitudeToDecibel, ApplyFilterbank
from ..main import parameters
import leaf_audio.frontend as leaf_frontend
from typing import Optional
from ..main import parameters


class leaf_model9(tf.keras.Model):
    """Neural network architecture to train an audio classifier from waveforms."""

    def __init__(self,
                 _frontend: Optional[tf.keras.Model] = None,
                 encoder: Optional[tf.keras.Model] = None,
                 ):
        """Initialization.
        Args:
          num_outputs: the number of classes of the classification problem.
          frontend: A keras model that takes a waveform and outputs a time-frequency
            representation.
          encoder: An encoder to turn the time-frequency representation into an
            embedding.
        """
        super().__init__()
        self._frontend = _frontend
        self._encoder = encoder
        KERNEL_SIZE = 6
        POOL_SIZE = (2, 2)

        self.bn1 = tf.keras.Sequential([
            BatchNormalization()
        ])
        self.tower_1 = tf.keras.Sequential([
            Conv2D(16, (1, 1), padding='same', activation='relu'),
            Conv2D(16, (3, 3), padding='same', activation='relu')
        ])
        self.tower_2 = tf.keras.Sequential([
            Conv2D(16, (1, 1), padding='same', activation='relu'),
            Conv2D(16, (5, 5), padding='same', activation='relu')
        ])
        self.tower_3 = tf.keras.Sequential([
            MaxPooling2D((3, 3), strides=(1, 1), padding='same'),
            Conv2D(16, (1, 1), padding='same', activation='relu')
        ])
        self.concat = Concatenate(axis=3)

        self._pool = tf.keras.Sequential([
            AveragePooling2D(pool_size=POOL_SIZE, padding="same"),
            BatchNormalization(),
            Dropout(0.1),
            InvertedResidual(filters=32, strides=1, kernel_size=KERNEL_SIZE),
            InvertedResidual(filters=32, strides=1, kernel_size=KERNEL_SIZE),
            AveragePooling2D(pool_size=POOL_SIZE),
            BatchNormalization(),
            Dropout(0.1),
            InvertedResidual(filters=64, strides=1, kernel_size=KERNEL_SIZE,),
            InvertedResidual(filters=64, strides=1, kernel_size=KERNEL_SIZE,),
            AveragePooling2D(pool_size=POOL_SIZE),
            BatchNormalization(),
            Dropout(0.1),
            InvertedResidual(filters=128, strides=1, kernel_size=KERNEL_SIZE,),
            InvertedResidual(filters=128, strides=1, kernel_size=KERNEL_SIZE,),
            AveragePooling2D(pool_size=POOL_SIZE),
            BatchNormalization(),
            Dropout(0.1),
            InvertedResidual(filters=256, strides=1, kernel_size=KERNEL_SIZE),
            InvertedResidual(filters=256, strides=1, kernel_size=KERNEL_SIZE),
            AveragePooling2D(pool_size=POOL_SIZE),
            BatchNormalization(),
            Dropout(0.1),
            InvertedResidual(filters=512, strides=1, kernel_size=KERNEL_SIZE),
            InvertedResidual(filters=512, strides=1, kernel_size=KERNEL_SIZE),
            GlobalAveragePooling2D(),
            Dense(parameters.n_classes, activity_regularizer=l2(
                parameters.ll2_reg), activation=parameters.activation)
        ])

    def call(self, inputs: tf.Tensor, training: bool = True, return_spec: bool = False):
        output = inputs
        if self._frontend is not None:
            output = self._frontend(
                output, training=training)  # pylint: disable=not-callable
            if parameters.normalize:
                output = output/tf.reduce_max(output)
                output = 2 * output - 1
            if return_spec:
                return output
            output = tf.expand_dims(output, -1)
        if self._encoder:
            output = self._encoder(output, training=training)
        output = tf.transpose(output, [0, 2, 1, 3])

        output = self.bn1(output)
        o_1 = self.tower_1(output)
        o_2 = self.tower_2(output)
        o_3 = self.tower_3(output)
        output = self.concat([o_1, o_2, o_3])
        output = self._pool(output)
        return output

    def save(self, dest, epoch):
        print(type(self._frontend))
        try:
            # Raises an AttributeError
            self._frontend.save_weights(
                dest + "/_frontend_{}.h5".format(epoch))
        except AttributeError:
            print("There is no such attribute")
        self.bn1.save_weights(dest + "/bn1_{}.h5".format(epoch))
        self.tower_1.save_weights(dest + "/tower_1_{}.h5".format(epoch))
        self.tower_2.save_weights(dest + "/tower_2_{}.h5".format(epoch))
        self.tower_3.save_weights(dest + "/tower_3_{}.h5".format(epoch))
        self._pool.save_weights(dest + "/_pool_{}.h5".format(epoch))

    def _load(self, source, epoch):
        try:
            # Raises an AttributeError
            self._frontend.load_weights(
                source + "_frontend_{}.h5".format(epoch))
        except AttributeError:
            print("There is no such attribute")

        self.bn1.load_weights(source + "bn1_{}.h5".format(epoch))
        self.tower_1.load_weights(source + "tower_1_{}.h5".format(epoch))
        self.tower_2.load_weights(source + "tower_2_{}.h5".format(epoch))
        self.tower_3.load_weights(source + "tower_3_{}.h5".format(epoch))
        self._pool.load_weights(source + "_pool_{}.h5".format(epoch))

