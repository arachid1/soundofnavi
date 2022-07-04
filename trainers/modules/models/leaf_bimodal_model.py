import pandas as pd
import numpy as np
import tensorflow as tf
from .core import *
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from kapre.composed import get_melspectrogram_layer
from kapre.time_frequency import STFT, Magnitude, Phase, MagnitudeToDecibel, ApplyFilterbank
from ..main import parameters
import leaf_audio.frontend as frontend
from typing import Optional


class leaf_mixednet_model(tf.keras.Model):
  """Neural network architecture to train an audio classifier from waveforms."""

  def __init__(self,
               num_outputs: int,
               frontend: Optional[tf.keras.Model] = None,
               encoder: Optional[tf.keras.Model] = None):
    """Initialization.
    Args:
      num_outputs: the number of classes of the classification problem.
      frontend: A keras model that takes a waveform and outputs a time-frequency
        representation.
      encoder: An encoder to turn the time-frequency representation into an
        embedding.
    """
    super().__init__()
    self._frontend = frontend
    self._encoder = encoder
    KERNELS = [3, 4, 5, 6, 7]
    POOL_SIZE=2
    self._pool = tf.keras.Sequential([
        layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
        MixDepthGroupConvolution2D(kernels=KERNELS),
        layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
        MixDepthGroupConvolution2D(kernels=KERNELS),
        layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        MixDepthGroupConvolution2D(kernels=KERNELS),
        layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        MixDepthGroupConvolution2D(kernels=KERNELS),
        layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        MixDepthGroupConvolution2D(kernels=KERNELS),
        layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        # tf.keras.layers.GlobalMaxPooling2D(),
        # tf.keras.layers.Flatten(),
    ])
    self._head = layers.Dense(num_outputs, activity_regularizer=l2(parameters.ll2_reg), activation="sigmoid")

  def call(self, inputs: tf.Tensor, training: bool = True, return_spec: bool = False):
    output = inputs
    if self._frontend is not None:
      output = self._frontend(output, training=training)  # pylint: disable=not-callable
      if return_spec:
          return output
      output = tf.expand_dims(output, -1)
    if self._encoder:
      output = self._encoder(output, training=training)
    output = tf.transpose(output, [0, 2, 1, 3])
    output = self._pool(output)
    return self._head(output)