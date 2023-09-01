import pandas as pd
import numpy as np
import tensorflow as tf
from .core import *
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

# from kapre.composed import get_melspectrogram_layer
# from kapre.time_frequency import STFT, Magnitude, Phase, MagnitudeToDecibel, ApplyFilterbank
from ..main import parameters

# import leaf_audio.frontend as leaf_frontend
from typing import Optional
from ..main import parameters


class default(tf.keras.Model):
    """Neural network architecture to train an audio classifier from waveforms."""

    def __init__(
        self,
        num_outputs: int,
        _frontend: Optional[tf.keras.Model] = None,
        encoder: Optional[tf.keras.Model] = None,
        normalize: bool = True,
        weights: list = [],
        first: int = 64,
        second: int = 64,
        activate_weights: bool = True,
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
        # self._encoder = encoder
        # self.mel_loss = tf.Variable(0.5, trainable=True)
        # self.sinc_loss = tf.Variable(0.5, trainable=True)
        # self.first = first
        # self.second = first
        self.normalize = normalize
        self.activate_weights = activate_weights
        self.w = weights

        KERNEL_SIZE = 6
        POOL_SIZE = (2, 2)
        self.bce = tf.keras.losses.BinaryCrossentropy()

        self._pool = tf.keras.Sequential(
            [
                tf.keras.applications.resnet50.ResNet50(
                    include_top=False,
                    weights="imagenet",
                    input_tensor=None,
                    pooling=None,
                    classes=num_outputs,
                ),
                layers.GlobalAveragePooling2D(),
                # tf.keras.layers.Dense(),
                layers.Dense(
                    num_outputs,
                    activity_regularizer=tf.keras.regularizers.l2(parameters.ll2_reg),
                    activation=parameters.activation,
                ),
            ]
        )

    def compute_mr_spectral_loss(self, x):
        spec = self(x, return_spec=True)
        mel_spec = self._mel_frontend(x)
        sinc_spec = self._sinc_frontend(x)
        if parameters.normalize:
            mel_spec = mel_spec - tf.reduce_min(mel_spec)
            mel_spec = mel_spec / tf.reduce_max(mel_spec)
            mel_spec = mel_spec * 2 - 1
            sinc_spec = sinc_spec - tf.reduce_min(sinc_spec)
            sinc_spec = sinc_spec / tf.reduce_max(sinc_spec)
            sinc_spec = sinc_spec * 2 - 1

        # loss = tf.reduce_mean(tf.math.square(mel_spec-spec))
        # # tf.print(loss)
        # loss += tf.reduce_mean(tf.math.square(sinc_spec-spec))

        loss = tf.reduce_mean(tf.math.square(mel_spec - spec))
        # tf.print(loss)
        loss += tf.reduce_mean(tf.math.square(sinc_spec - spec))
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
            batch_weights = tf.map_fn(
                lambda _y: tf.gather(_y, 1) + tf.math.reduce_sum(_y), y
            )
            batch_weights = tf.map_fn(
                fn=lambda t: tf.gather(self.w, tf.cast(t, tf.int32)),
                elems=batch_weights,
            )
            sample_weight = batch_weights
            # tf.print(sample_weight)

        loss = self.compiled_loss(
            y, y_pred, sample_weight, regularization_losses=self.losses
        )

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
            output = self._frontend(
                output, training=training
            )  # pylint: disable=not-callable
            if parameters.normalize:
                output = output / tf.reduce_max(output)
                output = 2 * output - 1
            if return_spec:
                return output
            output = tf.expand_dims(output, -1)
        if self._encoder:
            output = self._encoder(output, training=training)
        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.repeat(output, repeats=[3], axis=3)
        output = self._pool(output)
        # tf.print(tf.shape(output))
        # output = self._final(output)
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
        # self.bn1.save_weights(dest + "/bn1_{}.h5".format(epoch))
        # self.tower_1.save_weights(dest + "/tower_1_{}.h5".format(epoch))
        # self.tower_2.save_weights(dest + "/tower_2_{}.h5".format(epoch))
        # self.tower_3.save_weights(dest + "/tower_3_{}.h5".format(epoch))
        self._pool.save_weights(dest + "/_pool_{}.h5".format(epoch))
        # self._final.save_weights(dest + "/_final_{}.h5".format(epoch))

    def _load(self, source, epoch):
        try:
            # Raises an AttributeError
            self._frontend.load_weights(source + "_frontend_{}.h5".format(epoch))
        except AttributeError:
            print("There is no such attribute")
        # self.bn1.load_weights(source + "bn1_{}.h5".format(epoch))
        # self.tower_1.load_weights(source + "tower_1_{}.h5".format(epoch))
        # self.tower_2.load_weights(source + "tower_2_{}.h5".format(epoch))
        # self.tower_3.load_weights(source + "tower_3_{}.h5".format(epoch))
        self._pool.load_weights(source + "_pool_{}.h5".format(epoch))
        # self._final.load_weights(source + "_final_{}.h5".format(epoch))


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
