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
               _frontend: Optional[tf.keras.Model] = None,
               encoder: Optional[tf.keras.Model] = None,
               normalize: bool = True,
               weights: list = [],
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
    self.normalize = normalize
    self.w = weights
    KERNELS = [3, 4, 5, 6, 7]
    POOL_SIZE=2
    self._functional = tf.keras.Sequential([
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
        layers.Dense(num_outputs, activity_regularizer=l2(parameters.ll2_reg), activation="sigmoid")
        # tf.keras.layers.GlobalMaxPooling2D(),
        # tf.keras.layers.Flatten(),
    ])
    # self._head = layers.Dense(num_outputs, activity_regularizer=l2(parameters.ll2_reg), activation="sigmoid")

  def call(self, inputs: tf.Tensor, training: bool = True, return_spec: bool = False):
    output = inputs
    if self._frontend is not None:
      output = self._frontend(output, training=training)  # pylint: disable=not-callable
      if self.normalize:
          output = output/tf.reduce_max(output)
          output = 2 * output - 1
      if return_spec:
          return output
      output = tf.expand_dims(output, -1)
    if self._encoder:
      output = self._encoder(output, training=training)
    if not self._encoder:
        output = tf.transpose(output, [0, 2, 1, 3])
    else:
        output = tf.expand_dims(output, axis=3)
    return self._functional(output)
    # return self._head(output)

  def compute_loss(self, x, y, y_pred, sample_weight):
      batch_weights = []
      batch_weights = tf.map_fn(lambda _y: tf.gather(_y, 1) + tf.math.reduce_sum(_y), y)
      batch_weights = tf.map_fn(fn=lambda t: tf.gather(self.w, tf.cast(t, tf.int32)), elems=batch_weights)
      loss = self.compiled_loss(y, y_pred, batch_weights, regularization_losses=self.losses)
    #   tf.print("here")
    #   tf.print(loss)
      # if self.activate_spectral_loss:
      #   loss += self.compute_mr_spectral_loss(x)
    #   tf.print(loss)
      return loss

  def save(self, dest, epoch):
    self._frontend.save_weights(dest + "/_frontend_{}.h5".format(epoch))
    self._functional.save_weights(dest + "/functional_{}.h5".format(epoch))

  def _load(self, source, epoch):
    self._frontend.load_weights(source + "_frontend_{}.h5".format(epoch))
    self._functional.load_wesights(source + "functional_{}.h5".format(epoch))


# class leaf_mixednet_model(tf.keras.Model):
#   """Neural network architecture to train an audio classifier from waveforms."""

#   def __init__(self,
#                num_outputs: int,
#                frontend: Optional[tf.keras.Model] = None,
#                encoder: Optional[tf.keras.Model] = None,
#                normalize: bool = True):
#     """Initialization.
#     Args:
#       num_outputs: the number of classes of the classification problem.
#       frontend: A keras model that takes a waveform and outputs a time-frequency
#         representation.
#       encoder: An encoder to turn the time-frequency representation into an
#         embedding.
#     """
#     super().__init__()
#     self._frontend = frontend
#     self._encoder = encoder
#     self.normalize = normalize
#     KERNELS = [3, 4, 5, 6, 7]
#     POOL_SIZE=2
#     self._functional = tf.keras.Sequential([
#         layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
#         MixDepthGroupConvolution2D(kernels=KERNELS),
#         layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same"),
#         layers.BatchNormalization(),
#         layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
#         MixDepthGroupConvolution2D(kernels=KERNELS),
#         layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same"),
#         layers.BatchNormalization(),
#         layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
#         MixDepthGroupConvolution2D(kernels=KERNELS),
#         layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same"),
#         layers.BatchNormalization(),
#         layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
#         MixDepthGroupConvolution2D(kernels=KERNELS),
#         layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same"),
#         layers.BatchNormalization(),
#         layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
#         MixDepthGroupConvolution2D(kernels=KERNELS),
#         layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same"),
#         layers.BatchNormalization(),
#         layers.GlobalAveragePooling2D(),
#         layers.Dense(num_outputs, activity_regularizer=l2(parameters.ll2_reg), activation="sigmoid")
#         # tf.keras.layers.GlobalMaxPooling2D(),
#         # tf.keras.layers.Flatten(),
#     ])
#     # self._head = layers.Dense(num_outputs, activity_regularizer=l2(parameters.ll2_reg), activation="sigmoid")

#   def call(self, inputs: tf.Tensor, training: bool = True, return_spec: bool = False):
#     output = inputs
#     if self._frontend is not None:
#       output = self._frontend(output, training=training)  # pylint: disable=not-callable
#       if self.normalize:
#           output = output/tf.reduce_max(output)
#           output = 2 * output - 1
#       if return_spec:
#           return output
#       output = tf.expand_dims(output, -1)
#     if self._encoder:
#       output = self._encoder(output, training=training)
#     output = tf.transpose(output, [0, 2, 1, 3])
#     return self._functional(output)
#     # return self._head(output)