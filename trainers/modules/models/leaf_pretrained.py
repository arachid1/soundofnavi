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


class sleaf_pretrained(tf.keras.Model):
    """Neural network architecture to train an audio classifier from waveforms."""

    def __init__(self,
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
        self.input_layer = tf.keras.layers.Input(shape=parameters.audio_shape)
        self._frontend = _frontend
        self._encoder = encoder
        self.normalize = normalize
        if parameters.model == "resnet":
            model = tf.keras.applications.resnet50.ResNet50(
                include_top=False,
                weights="imagenet",
                input_tensor=None,
                input_shape=parameters.spec_shape,
            )
        elif parameters.model == "effnet":
            model = tf.keras.applications.EfficientNetB1(
                include_top=False,
                weights="imagenet",
                input_tensor=None,
                input_shape=parameters.spec_shape,
            )

        self._pool = tf.keras.Sequential([
            model,
            layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(50, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            layers.Dense(parameters.n_classes, activity_regularizer=tf.keras.regularizers.l2(
                parameters.ll2_reg), activation=parameters.activation)
        ])

    def analyze_layers(self, domain_examples):

        self.focus_layers = [self.get_layer('leaf'), self.get_layer(
            'sequential').get_layer(
            'resnet50').get_layer(
            'conv1_relu'), self.get_layer('sequential').get_layer('dense')]
        for (x, y) in domain_examples:
            for l in self.focus_layers:
                grad_model = tf.keras.models.Model(
                    [self.inputs], [l.output, self.output]
                )
                activation = grad_model(x, training=False)
                print(activation)
                exit()

        return None
        # frontend part

    def compute_weights(self, y):
        sample_weight = None
        if parameters.use_class_weights:
            sample_weight = tf.map_fn(lambda _y: tf.gather(
                _y, 1) + tf.math.reduce_sum(_y), y)
            sample_weight = tf.map_fn(fn=lambda t: tf.gather(
                parameters.weights, tf.cast(t, tf.int32)), dtype=tf.float64, elems=sample_weight)
        return sample_weight

    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            sample_weight = self.compute_weights(y)
            loss_value = self.compute_loss(
                x, y, y_pred, sample_weight=sample_weight)
            loss_value += sum(self.losses)
        grads = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        for m in self.compiled_metrics._metrics:
            m.update_state(y, y_pred)
        # self.compiled_metrics.update_state(y, y_pred)
        # metrics_results = self.compute_metrics(
        #     x, y, y_pred, sample_weight=sample_weight)
        # print(metrics_results)
        return loss_value

    def test_step(self, x, y):
        y_pred = self(x, training=False)
        sample_weight = self.compute_weights(y)
        loss_value = self.compute_loss(
            x, y, y_pred, sample_weight=sample_weight)
        loss_value += sum(self.losses)
        for m in self.compiled_metrics._metrics:
            m.update_state(y, y_pred)
        return loss_value

    def build(self, input_shape):
        self._pool(self._frontend(self.input_layer))

    def call(self, inputs, training=True, return_spec=False):

        # output = self.input_layer(inputs)
        output = inputs
        if self._frontend is not None:
            output = self._frontend(
                output, training=training)  # pylint: disable=not-callable
            if parameters.normalize:
                output = output - tf.math.reduce_min(output)
                output = output/tf.math.reduce_max(output)
            if return_spec:
                return output
            output = tf.expand_dims(output, -1)
        if self._encoder:
            output = self._encoder(output, training=training)

        output = tf.transpose(output, [0, 2, 1, 3])
        # if parameters.stacking:
        #     mel = self._mel_frontend(inputs)
        #     mel = mel - tf.math.reduce_min(mel)
        #     mel = mel/tf.math.reduce_max(mel)
        #     mel = 2 * mel - 1
        #     mel = tf.expand_dims(mel, -1)
        #     mel = tf.transpose(mel, [0, 2, 1, 3])

        #     sinc = self._sinc_frontend(inputs)
        #     sinc = sinc - tf.math.reduce_min(sinc)
        #     sinc = sinc/tf.math.reduce_max(sinc)
        #     sinc = 2 * sinc - 1
        #     sinc = tf.expand_dims(sinc, -1)
        #     sinc = tf.transpose(sinc, [0, 2, 1, 3])
        #     output = tf.concat([output, mel, sinc], axis=3)
        output = tf.repeat(output, repeats=[3], axis=3)
        output = self._pool(output)
        # if parameters.distill_features:
        #   output = tf.math.log_sigmoid(output)
        return output

    def save(self, dest, epoch):
        try:
            # Raises an AttributeError
            self._frontend.save_weights(
                dest + "/_frontend_{}.h5".format(epoch))
        except AttributeError:
            print("There is no such attribute")
        self._pool.save_weights(dest + "/_pool_{}.h5".format(epoch))

    def _load(self, source, epoch):
        try:
            # Raises an AttributeError
            self._frontend.load_weights(
                source + "_frontend_{}.h5".format(epoch))
        except AttributeError:
            print("There is no such attribute")
        self._pool.load_weights(source + "_pool_{}.h5".format(epoch))
