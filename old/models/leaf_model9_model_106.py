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


class spectral_loss_layer(tf.keras.layers.Layer):
  def __init__(self):
    super(spectral_loss_layer, sample_rate, n_filters, self).__init__()
    # self.num_outputs = num_outputs
    self._mel_frontend = leaf_frontend.MelFilterbanks(sample_rate=sample_rate, n_filters=n_filters)
    self._sinc_frontend = leaf_frontend.SincNet(sample_rate=sample_rate, n_filters=n_filters)

  def build(self, input_shape):
    self.mel_loss = tf.Variable(0.5, trainable=True)
    self.sinc_loss = tf.Variable(0.5, trainable=True)
    # self.kernel = self.add_weight("kernel",
    #                               shape=[int(input_shape[-1]),
    #                                      self.num_outputs])

  def call(self, inputs):
    # spec = self(x, return_spec=True)
    spec = inputs
    mel_spec = self._mel_frontend(x)
    sinc_spec = self._sinc_frontend(x)
    if self.normalize:
        mel_spec = mel_spec - tf.reduce_min(mel_spec)
        mel_spec = mel_spec / tf.reduce_max(mel_spec)
        mel_spec = mel_spec*2-1
        sinc_spec = sinc_spec - tf.reduce_min(sinc_spec)
        sinc_spec = sinc_spec / tf.reduce_max(sinc_spec)
        sinc_spec = sinc_spec*2-1

    loss = self.mel_loss*tf.reduce_mean(tf.math.square(mel_spec-spec))
    # tf.print(loss)
    loss += self.sinc_loss* tf.reduce_mean(tf.math.square(sinc_spec-spec))
    # tf.summary.scalar("spectral loss", loss, step=)
    # self.mr_loss = loss

    return loss




class leaf_model9_model_106(tf.keras.Model):
  """Neural network architecture to train an audio classifier from waveforms."""

  def __init__(self,
               num_outputs: int,
               _frontend: Optional[tf.keras.Model] = None,
               encoder: Optional[tf.keras.Model] = None,
               normalize: bool = True,
               weights: list = [], 
               first: int = 64, 
               second: int = 64,
               activate_weights: bool = True
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
    self._mel_frontend = leaf_frontend.MelFilterbanks(sample_rate=parameters.sr, n_filters=parameters.n_filters)
    self._sinc_frontend = leaf_frontend.SincNet(sample_rate=parameters.sr, n_filters=parameters.n_filters)
    # self.loss_layer = spectral_loss_layer()
    self._encoder = encoder
    self.mel_loss = tf.Variable(0.5, trainable=True)
    self.sinc_loss = tf.Variable(0.5, trainable=True)

    self.first = first
    self.second = first
    self.normalize = normalize
    self.activate_weights = activate_weights
    self.w = weights

    KERNEL_SIZE = 6
    POOL_SIZE = (2, 2)
    self.bce = tf.keras.losses.BinaryCrossentropy()

    self.bn1 = tf.keras.Sequential([
        layers.BatchNormalization()
    ])
    self.tower_1 = tf.keras.Sequential([
        layers.Conv2D(16, (1,1), padding='same', activation='relu'),
        layers.Conv2D(16, (3,3), padding='same', activation='relu')
    ])
    self.tower_2 = tf.keras.Sequential([
        layers.Conv2D(16, (1,1), padding='same', activation='relu'),
        layers.Conv2D(16, (5,5), padding='same', activation='relu')
    ])
    self.tower_3 = tf.keras.Sequential([
        layers.MaxPooling2D((3,3), strides=(1,1), padding='same'),
        layers.Conv2D(16, (1,1), padding='same', activation='relu')
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
        # layers.TimeDistributed(layers.GlobalAveragePooling2D())
        # layers.GlobalAveragePooling2D(),
        layers.Reshape((15, 2*512)),
    ])
    self._final = tf.keras.Sequential([
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(self.second, return_sequences=True)),
        layers.Dense(100, activation="relu"),
        layers.Dropout(0.5),
        # layers.Dense(50, activation="relu"),
        # layers.Dropout(0.5),
        # layers.Dense(num_outputs, activity_regularizer=l2(parameters.ll2_reg), activation="sigmoid")
        layers.Flatten(),
        layers.Dense(num_outputs, activity_regularizer=l2(parameters.ll2_reg), activation=parameters.activation)
  
    ])


  def compute_mr_spectral_loss(self, x):
      spec = self(x, return_spec=True)
      mel_spec = self._mel_frontend(x)
      sinc_spec = self._sinc_frontend(x)
      if parameters.normalize:
          mel_spec = mel_spec - tf.reduce_min(mel_spec)
          mel_spec = mel_spec / tf.reduce_max(mel_spec)
          mel_spec = mel_spec*2-1
          sinc_spec = sinc_spec - tf.reduce_min(sinc_spec)
          sinc_spec = sinc_spec / tf.reduce_max(sinc_spec)
          sinc_spec = sinc_spec*2-1

      # loss = tf.reduce_mean(tf.math.square(mel_spec-spec))
      # # tf.print(loss)
      # loss += tf.reduce_mean(tf.math.square(sinc_spec-spec))

      loss = tf.reduce_mean(tf.math.square(mel_spec-spec))
      # tf.print(loss)
      loss +=  tf.reduce_mean(tf.math.square(sinc_spec-spec))
      # tf.summary.scalar("spectral loss", loss, step=)
      # self.mr_loss = loss

      return loss

  def compute_loss(self, x, y, y_pred, sample_weight):

      # print(step)
      # exit()

      if parameters.class_weights:
        # tf.print("true")
        # tf.print(y)
        # tf.print(sample_weight)
        batch_weights = []
        batch_weights = tf.map_fn(lambda _y: tf.gather(_y, 1) + tf.math.reduce_sum(_y), y)
        batch_weights = tf.map_fn(fn=lambda t: tf.gather(self.w, tf.cast(t, tf.int32)), elems=batch_weights)
        sample_weight = batch_weights
        # tf.print(sample_weight)

      
      loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
        
      # loss = loss_layer(x)
      if parameters.activate_spectral_loss:
        loss += self.compute_mr_spectral_loss(x)

      # for label in y:
      #   tf.print(label)
      #   a = tf.equal(label, [0,0])
      #   tf.print(a)
      #   a = tf.math.count_nonzero(a)
      #   tf.print(a)
      # if y == np.array([0,0]):
      #   loss = loss * self.w[0]
      # elif y == [1,0]:
      #   loss = loss * self.w[1]
      # elif y == [0,1]:
      #   loss = loss * self.w[2]
      # else:
      #   loss = loss * self.w[3]
      # tf.print(loss)
      return loss

  def call(self, inputs: tf.Tensor, training: bool = True, return_spec: bool = False):
    output = inputs
    if self._frontend is not None:
      output = self._frontend(output, training=training)  # pylint: disable=not-callable
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
    output = self.concat([o_1,o_2,o_3])
    output = self._pool(output)
    # tf.print(tf.shape(output))
    output = self._final(output)
    # tf.print(tf.shape(output))
    # print(tf.shape(output))
    # exit()
    # output = self.last(output)
    # tf.print(tf.shape(output))
    return output

  def save(self, dest, epoch):
    try:
      # Raises an AttributeError
      self._frontend.save_weights(dest + "/_frontend_{}.h5".format(epoch))
    except AttributeError:
        print("There is no such attribute")
    self.bn1.save_weights(dest + "/bn1_{}.h5".format(epoch))
    self.tower_1.save_weights(dest + "/tower_1_{}.h5".format(epoch))
    self.tower_2.save_weights(dest + "/tower_2_{}.h5".format(epoch))
    self.tower_3.save_weights(dest + "/tower_3_{}.h5".format(epoch))
    self._pool.save_weights(dest + "/_pool_{}.h5".format(epoch))
    self._final.save_weights(dest + "/_final_{}.h5".format(epoch))

  def _load(self, source, epoch):
    try:
      # Raises an AttributeError
      self._frontend.load_weights(source + "_frontend_{}.h5".format(epoch))
    except AttributeError:
        print("There is no such attribute")
    self.bn1.load_weights(source + "bn1_{}.h5".format(epoch))
    self.tower_1.load_weights(source + "tower_1_{}.h5".format(epoch))
    self.tower_2.load_weights(source + "tower_2_{}.h5".format(epoch))
    self.tower_3.load_weights(source + "tower_3_{}.h5".format(epoch))
    self._pool.load_weights(source + "_pool_{}.h5".format(epoch))
    self._final.load_weights(source + "_final_{}.h5".format(epoch))




    
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