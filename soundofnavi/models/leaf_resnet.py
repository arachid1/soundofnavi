import pandas as pd
import numpy as np
import tensorflow as tf
from .core import *
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from kapre.composed import get_melspectrogram_layer
from kapre.time_frequency import STFT, Magnitude, Phase, MagnitudeToDecibel, ApplyFilterbank
from ..main import parameters
import leaf_audio.frontend as leaf_frontend
from typing import Optional
from ..main import parameters

class leaf_resnet(tf.keras.Model):
  """Neural network architecture to train an audio classifier from waveforms."""

  def __init__(self,
               num_outputs: int,
               _frontend: Optional[tf.keras.Model] = None,
               encoder: Optional[tf.keras.Model] = None,
               normalize: bool = True):
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
    self._mel_frontend = leaf_frontend.MelFilterbanks(sample_rate=self._frontend.sample_rate, n_filters=self._frontend.n_filters)
    self._sinc_frontend = leaf_frontend.SincNet(sample_rate=self._frontend.sample_rate, n_filters=self._frontend.n_filters)
    self._leaf_frontend_bis = leaf_frontend.Leaf(sample_rate=8000, n_filters=80, window_stride=20)
    self._encoder = encoder
    self.normalize = normalize
    KERNEL_SIZE = 6
    POOL_SIZE = (2, 2)
    self.bce = tf.keras.losses.BinaryCrossentropy()
    self._pool =  tf.keras.Sequential([
        tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(80,500,3),
        pooling=None,
        classes=num_outputs),
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_outputs, activity_regularizer=l2(parameters.ll2_reg), activation="sigmoid")
    ])
    # self.final = layers.Dense(num_outputs, activity_regularizer=l2(parameters.ll2_reg), activation="sigmoid")
        
    # ])

    # self.bn1 = tf.keras.Sequential([
    #     layers.BatchNormalization()
    # ])
    # self.tower_1 = tf.keras.Sequential([
    #     layers.Conv2D(16, (1,1), padding='same', activation='relu'),
    #     layers.Conv2D(16, (3,3), padding='same', activation='relu')
    # ])
    # self.tower_2 = tf.keras.Sequential([
    #     layers.Conv2D(16, (1,1), padding='same', activation='relu'),
    #     layers.Conv2D(16, (5,5), padding='same', activation='relu')
    # ])
    # self.tower_3 = tf.keras.Sequential([
    #     layers.MaxPooling2D((3,3), strides=(1,1), padding='same'),
    #     layers.Conv2D(16, (1,1), padding='same', activation='relu')
    # ])
    # self.concat = layers.Concatenate(axis=3)

    # self._pool = tf.keras.Sequential([
    #     layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same"),
    #     layers.BatchNormalization(),
    #     layers.Dropout(0.1),
    #     InvertedResidual(filters=32, strides=1, kernel_size=KERNEL_SIZE),
    #     InvertedResidual(filters=32, strides=1, kernel_size=KERNEL_SIZE),
    #     layers.AveragePooling2D(pool_size=POOL_SIZE),
    #     layers.BatchNormalization(),
    #     layers.Dropout(0.1),
    #     InvertedResidual(filters=64, strides=1, kernel_size=KERNEL_SIZE,),
    #     InvertedResidual(filters=64, strides=1, kernel_size=KERNEL_SIZE,),
    #     layers.AveragePooling2D(pool_size=POOL_SIZE),
    #     layers.BatchNormalization(),
    #     layers.Dropout(0.1),
    #     InvertedResidual(filters=128, strides=1, kernel_size=KERNEL_SIZE,),
    #     InvertedResidual(filters=128, strides=1, kernel_size=KERNEL_SIZE,),
    #     layers.AveragePooling2D(pool_size=POOL_SIZE),
    #     layers.BatchNormalization(),
    #     layers.Dropout(0.1),
    #     InvertedResidual(filters=256, strides=1, kernel_size=KERNEL_SIZE),
    #     InvertedResidual(filters=256, strides=1, kernel_size=KERNEL_SIZE),
    #     layers.AveragePooling2D(pool_size=POOL_SIZE),
    #     layers.BatchNormalization(),
    #     layers.Dropout(0.1),
    #     InvertedResidual(filters=512, strides=1, kernel_size=KERNEL_SIZE),
    #     InvertedResidual(filters=512, strides=1, kernel_size=KERNEL_SIZE),
    #     layers.GlobalAveragePooling2D(),
        # layers.Dense(num_outputs, activity_regularizer=l2(parameters.ll2_reg), activation="sigmoid")
    # ])


  def compute_mr_spectral_loss(self, x):
      spec = self(x, return_spec=True)
      mel_spec = self._mel_frontend(x)
      sinc_spec = self._sinc_frontend(x)
      leaf_bis = self._leaf_frontend_bis(x)
      if self.normalize:
          mel_spec = mel_spec - tf.reduce_min(mel_spec)
          mel_spec = mel_spec / tf.reduce_max(mel_spec)
          mel_spec = mel_spec*2-1
          sinc_spec = sinc_spec - tf.reduce_min(sinc_spec)
          sinc_spec = sinc_spec / tf.reduce_max(sinc_spec)
          sinc_spec = sinc_spec*2-1
          leaf_bis = leaf_bis - tf.reduce_min(leaf_bis)
          leaf_bis = leaf_bis / tf.reduce_max(leaf_bis)
          leaf_bis = leaf_bis*2-1

      loss = tf.reduce_mean(tf.math.square(mel_spec-spec))
      # tf.print(loss)
      loss += tf.reduce_mean(tf.math.square(sinc_spec-spec))
      loss += tf.reduce_mean(tf.math.square(leaf_bis-spec))
      # tf.print(loss)
      loss = tf.math.sqrt(loss)
  
      # tf.print(loss)

      return loss

  def compute_loss(self, x, y, y_pred, sample_weight):
      loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
      # tf.print("first")
      # tf.print(loss)
      # loss += self.compute_mr_spectral_loss(x)
    #   tf.print(loss)
      return loss

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
    output = tf.transpose(output, [0, 2, 1, 3])
    # tf.print(tf.shape(output))
    output = tf.repeat(output, repeats=[3], axis=3)
    # b = tf.tile(a, multiples)
    # tf.print(tf.shape(output))

    output = self._pool(output)
    # tf.print(tf.shape(output))
    # output = tf.keras.layers.Flatten()(output)
    # output = self.final(output)

    # output = self._dense(output)
    # tf.print(tf.shape(output))
    return output


  def save(self, dest, epoch):
    self._frontend.save_weights(dest + "/_frontend_{}.h5".format(epoch))
    self._pool.save_weights(dest + "/_pool{}.h5".format(epoch))


  def _load(self, source, epoch):
    self._frontend.load_weights(source + "_frontend_{}.h5".format(epoch))
    self._pool.load_weights(source + "_pool{}.h5".format(epoch))





    
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