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
from ..main import parameters

class leaf_MoE_model9_model(tf.keras.Model):
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
    KERNEL_SIZE = 6
    POOL_SIZE = (2, 2)
    # x = layers.BatchNormalization()(x)
    # tower_1 = tf.keras.Sequential([
    #     layers.Conv2D(16, (1,1), padding='same', activation='relu'),
    #     layers.Conv2D(16, (3,3), padding='same', activation='relu')
    # ])
    # tower_2 = tf.keras.Sequential([
    #     layers.Conv2D(16, (1,1), padding='same', activation='relu'),
    #     layers.Conv2D(16, (5,5), padding='same', activation='relu')
    # ])
    # tower_3 = tf.keras.Sequential([
    #     layers.MaxPooling2D((3,3), strides=(1,1), padding='same'),
    #     layers.Conv2D(16, (1,1), padding='same', activation='relu')
    # ])
    # self._pool = tf.keras.Sequential([
    #     layers.BatchNormalization(),
    #     layers.Concatenate([tower_1, tower_2, tower_3]),
    #     layers.AveragePooling2D(pool_size=self.POOL_SIZE, padding="same"),
    #     layers.BatchNormalization(),
    #     layers.Dropout(0.1),
    #     layers.GlobalAveragePooling2D()
    # ])
    self.bn1 = layers.BatchNormalization()
    self.tower_1 = tf.keras.Sequential([
        Conv2DMoE(8, 2, (1, 1), expert_activation='relu', padding='same', gating_activation='sigmoid'),
        Conv2DMoE(8, 2, (3, 3), expert_activation='relu', padding='same', gating_activation='sigmoid')
    ])
    self.tower_2 = tf.keras.Sequential([
        Conv2DMoE(8, 2, (1, 1), expert_activation='relu', padding='same', gating_activation='sigmoid'),
        Conv2DMoE(8, 2, (5, 5), expert_activation='relu', padding='same', gating_activation='sigmoid')
    ])
    self.tower_3 = tf.keras.Sequential([
        layers.MaxPooling2D((3,3), strides=(1,1), padding='same'),
        Conv2DMoE(8, 2, (1, 1), expert_activation='relu', padding='same', gating_activation='sigmoid')
    ])
    self.concat = layers.Concatenate(axis=3)

    self._pool = tf.keras.Sequential([
        layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same"),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        InvertedResidual(filters=32, strides=1, kernel_size=KERNEL_SIZE),
        InvertedResidual(filters=32, strides=1, kernel_size=KERNEL_SIZE),
        layers.AveragePooling2D(pool_size=POOL_SIZE),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        InvertedResidual(filters=64, strides=1, kernel_size=KERNEL_SIZE,),
        InvertedResidual(filters=64, strides=1, kernel_size=KERNEL_SIZE,),
        layers.AveragePooling2D(pool_size=POOL_SIZE),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        InvertedResidual(filters=128, strides=1, kernel_size=KERNEL_SIZE,),
        InvertedResidual(filters=128, strides=1, kernel_size=KERNEL_SIZE,),
        layers.AveragePooling2D(pool_size=POOL_SIZE),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        InvertedResidual(filters=256, strides=1, kernel_size=KERNEL_SIZE),
        InvertedResidual(filters=256, strides=1, kernel_size=KERNEL_SIZE),
        layers.AveragePooling2D(pool_size=POOL_SIZE),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        InvertedResidual(filters=512, strides=1, kernel_size=KERNEL_SIZE),
        InvertedResidual(filters=512, strides=1, kernel_size=KERNEL_SIZE),
        layers.GlobalAveragePooling2D()
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

    output = self.bn1(output)
    o_1 = self.tower_1(output)
    o_2 = self.tower_2(output)
    o_3 = self.tower_3(output)
    output = self.concat([o_1,o_2,o_3])
    output = self._pool(output)

    return self._head(output)
    
# def leaf_model9_model(frontend, opt):

#     KERNEL_SIZE = 6
#     POOL_SIZE = (2, 2)
#     i = layers.Input(shape=parameters.shape)
#     x = frontend(i)
#     if return_spec:
#         return output
#     x = layers.Reshape((-1, parameters.n_filters, 1))(x)
#     x = layers.BatchNormalization()(x)
#     tower_1 = layers.Conv2D(16, (1,1), padding='same', activation='relu')(x)
#     tower_1 = layers.Conv2D(16, (3,3), padding='same', activation='relu')(tower_1)
#     tower_2 = layers.Conv2D(16, (1,1), padding='same', activation='relu')(x)
#     tower_2 = layers.Conv2D(16, (5,5), padding='same', activation='relu')(tower_2)
#     tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
#     tower_3 = layers.Conv2D(16, (1,1), padding='same', activation='relu')(tower_3)
#     x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
#     x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(0.1)(x)
#     x = InvertedResidual(filters=32, strides=1, kernel_size=KERNEL_SIZE)(x)
#     x = InvertedResidual(filters=32, strides=1, kernel_size=KERNEL_SIZE)(x)
#     x = layers.AveragePooling2D(pool_size=POOL_SIZE)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(0.1)(x)
#     x = InvertedResidual(filters=64, strides=1, kernel_size=KERNEL_SIZE,)(x)
#     x = InvertedResidual(filters=64, strides=1, kernel_size=KERNEL_SIZE,)(x)
#     x = layers.AveragePooling2D(pool_size=POOL_SIZE)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(0.1)(x)
#     x = InvertedResidual(filters=128, strides=1, kernel_size=KERNEL_SIZE,)(x)
#     x = InvertedResidual(filters=128, strides=1, kernel_size=KERNEL_SIZE,)(x)
#     x = layers.AveragePooling2D(pool_size=POOL_SIZE)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(0.1)(x)
#     x = InvertedResidual(filters=256, strides=1, kernel_size=KERNEL_SIZE)(x)
#     x = InvertedResidual(filters=256, strides=1, kernel_size=KERNEL_SIZE)(x)
#     x = layers.AveragePooling2D(pool_size=POOL_SIZE)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(0.1)(x)
#     x = InvertedResidual(filters=512, strides=1, kernel_size=KERNEL_SIZE)(x)
#     x = InvertedResidual(filters=512, strides=1, kernel_size=KERNEL_SIZE)(x)
#     x = layers.GlobalAveragePooling2D()(x)
#     o = layers.Dense(parameters.n_classes, activity_regularizer=l2(
#         parameters.ll2_reg), activation="sigmoid")(x)
#     # delete above

#     model = Model(inputs=i, outputs=o, name="model9")
#     return model