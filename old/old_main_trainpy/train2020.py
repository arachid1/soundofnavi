from .modules.helpers import *
from .modules.generators import *
from .modules.metrics import *
from .modules.callbacks import *
from .core import *

import argparse
import pickle
import tensorflow as tf
import sys
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras import backend as K
from tensorflow.keras.initializers import RandomUniform
from keras import activations, initializers, regularizers, constraints
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
from sklearn.utils import class_weight

from tensorflow.python.lib.io import file_io
import json
import time
import numpy as np

class DenseMoE(layers.Layer):
    """Mixture-of-experts layer.
    Implements: y = sum_{k=1}^K g(v_k * x) f(W_k * x)
        # Arguments
        units: Positive integer, dimensionality of the output space.
        n_experts: Positive integer, number of experts (K).
        expert_activation: Activation function for the expert model (f).
        gating_activation: Activation function for the gating model (g).
        use_expert_bias: Boolean, whether to use biases in the expert model.
        use_gating_bias: Boolean, whether to use biases in the gating model.
        expert_kernel_initializer_scale: Float, scale of Glorot uniform initialization for expert model weights.
        gating_kernel_initializer_scale: Float, scale of Glorot uniform initialization for gating model weights.
        expert_bias_initializer: Initializer for the expert biases.
        gating_bias_initializer: Initializer fot the gating biases.
        expert_kernel_regularizer: Regularizer for the expert model weights.
        gating_kernel_regularizer: Regularizer for the gating model weights.
        expert_bias_regularizer: Regularizer for the expert model biases.
        gating_bias_regularizer: Regularizer for the gating model biases.
        expert_kernel_constraint: Constraints for the expert model weights.
        gating_kernel_constraint: Constraints for the gating model weights.
        expert_bias_constraint: Constraints for the expert model biases.
        gating_bias_constraint: Constraints for the gating model biases.
        activity_regularizer: Activity regularizer.
    # Input shape
        nD tensor with shape: (batch_size, ..., input_dim).
        The most common situation would be a 2D input with shape (batch_size, input_dim).
    # Output shape
        nD tensor with shape: (batch_size, ..., units).
        For example, for a 2D input with shape (batch_size, input_dim), the output would have shape (batch_size, units).
    """
    def __init__(self, units,
                 n_experts,
                 expert_activation=None,
                 gating_activation=None,
                 use_expert_bias=True,
                 use_gating_bias=True,
                 expert_kernel_initializer_scale=1.0,
                 gating_kernel_initializer_scale=1.0,
                 expert_bias_initializer='zeros',
                 gating_bias_initializer='zeros',
                 expert_kernel_regularizer=None,
                 gating_kernel_regularizer=None,
                 expert_bias_regularizer=None,
                 gating_bias_regularizer=None,
                 expert_kernel_constraint=None,
                 gating_kernel_constraint=None,
                 expert_bias_constraint=None,
                 gating_bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DenseMoE, self).__init__(**kwargs)
        self.units = units
        self.n_experts = n_experts

        self.expert_activation = activations.get(expert_activation)
        self.gating_activation = activations.get(gating_activation)

        self.use_expert_bias = use_expert_bias
        self.use_gating_bias = use_gating_bias

        self.expert_kernel_initializer_scale = expert_kernel_initializer_scale
        self.gating_kernel_initializer_scale = gating_kernel_initializer_scale

        self.expert_bias_initializer = initializers.get(expert_bias_initializer)
        self.gating_bias_initializer = initializers.get(gating_bias_initializer)

        self.expert_kernel_regularizer = regularizers.get(expert_kernel_regularizer)
        self.gating_kernel_regularizer = regularizers.get(gating_kernel_regularizer)

        self.expert_bias_regularizer = regularizers.get(expert_bias_regularizer)
        self.gating_bias_regularizer = regularizers.get(gating_bias_regularizer)

        self.expert_kernel_constraint = constraints.get(expert_kernel_constraint)
        self.gating_kernel_constraint = constraints.get(gating_kernel_constraint)

        self.expert_bias_constraint = constraints.get(expert_bias_constraint)
        self.gating_bias_constraint = constraints.get(gating_bias_constraint)

        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = layers.InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        expert_init_lim = np.sqrt(3.0*self.expert_kernel_initializer_scale / (max(1., float(input_dim + self.units) / 2)))
        gating_init_lim = np.sqrt(3.0*self.gating_kernel_initializer_scale / (max(1., float(input_dim + 1) / 2)))

        self.expert_kernel = self.add_weight(shape=(input_dim, self.units, self.n_experts),
                                      initializer=RandomUniform(minval=-expert_init_lim,maxval=expert_init_lim),
                                      name='expert_kernel',
                                      regularizer=self.expert_kernel_regularizer,
                                      constraint=self.expert_kernel_constraint)

        self.gating_kernel = self.add_weight(shape=(input_dim, self.n_experts),
                                      initializer=RandomUniform(minval=-gating_init_lim,maxval=gating_init_lim),
                                      name='gating_kernel',
                                      regularizer=self.gating_kernel_regularizer,
                                      constraint=self.gating_kernel_constraint)

        if self.use_expert_bias:
            self.expert_bias = self.add_weight(shape=(self.units, self.n_experts),
                                        initializer=self.expert_bias_initializer,
                                        name='expert_bias',
                                        regularizer=self.expert_bias_regularizer,
                                        constraint=self.expert_bias_constraint)
        else:
            self.expert_bias = None

        if self.use_gating_bias:
            self.gating_bias = self.add_weight(shape=(self.n_experts,),
                                        initializer=self.gating_bias_initializer,
                                        name='gating_bias',
                                        regularizer=self.gating_bias_regularizer,
                                        constraint=self.gating_bias_constraint)
        else:
            self.gating_bias = None

        self.input_spec = layers.InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):

        expert_outputs = tf.tensordot(inputs, self.expert_kernel, axes=1)
        if self.use_expert_bias:
            expert_outputs = K.bias_add(expert_outputs, self.expert_bias)
        if self.expert_activation is not None:
            expert_outputs = self.expert_activation(expert_outputs)

        gating_outputs = K.dot(inputs, self.gating_kernel)
        if self.use_gating_bias:
            gating_outputs = K.bias_add(gating_outputs, self.gating_bias)
        if self.gating_activation is not None:
            gating_outputs = self.gating_activation(gating_outputs)

        output = K.sum(expert_outputs * K.repeat_elements(K.expand_dims(gating_outputs, axis=1), self.units, axis=1), axis=2)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'n_experts':self.n_experts,
            'expert_activation': activations.serialize(self.expert_activation),
            'gating_activation': activations.serialize(self.gating_activation),
            'use_expert_bias': self.use_expert_bias,
            'use_gating_bias': self.use_gating_bias,
            'expert_kernel_initializer_scale': self.expert_kernel_initializer_scale,
            'gating_kernel_initializer_scale': self.gating_kernel_initializer_scale,
            'expert_bias_initializer': initializers.serialize(self.expert_bias_initializer),
            'gating_bias_initializer': initializers.serialize(self.gating_bias_initializer),
            'expert_kernel_regularizer': regularizers.serialize(self.expert_kernel_regularizer),
            'gating_kernel_regularizer': regularizers.serialize(self.gating_kernel_regularizer),
            'expert_bias_regularizer': regularizers.serialize(self.expert_bias_regularizer),
            'gating_bias_regularizer': regularizers.serialize(self.gating_bias_regularizer),
            'expert_kernel_constraint': constraints.serialize(self.expert_kernel_constraint),
            'gating_kernel_constraint': constraints.serialize(self.gating_kernel_constraint),
            'expert_bias_constraint': constraints.serialize(self.expert_bias_constraint),
            'gating_bias_constraint': constraints.serialize(self.gating_bias_constraint),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer)
        }
        base_config = super(DenseMoE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def conv2d(N_CLASSES, SR, BATCH_SIZE, LR, SHAPE, WEIGHT_DECAY, LL2_REG, EPSILON, LABEL_SMOOTHING):

    KERNEL_SIZE = 6
    POOL_SIZE = (2, 2)
    i = layers.Input(shape=SHAPE, )
    x = layers.BatchNormalization()(i)
    tower_1 = layers.Conv2D(16, (1,1), padding='same', activation='relu')(x)
    tower_1 = layers.Conv2D(16, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = layers.Conv2D(16, (1,1), padding='same', activation='relu')(x)
    tower_2 = layers.Conv2D(16, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    tower_3 = layers.Conv2D(16, (1,1), padding='same', activation='relu')(tower_3)
    x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = InvertedResidual(filters=32, strides=1, kernel_size=KERNEL_SIZE)(x)
    x = InvertedResidual(filters=32, strides=1, kernel_size=KERNEL_SIZE)(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = InvertedResidual(filters=64, strides=1, kernel_size=KERNEL_SIZE,)(x)
    x = InvertedResidual(filters=64, strides=1, kernel_size=KERNEL_SIZE,)(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = InvertedResidual(filters=128, strides=1, kernel_size=KERNEL_SIZE,)(x)
    x = InvertedResidual(filters=128, strides=1, kernel_size=KERNEL_SIZE,)(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = InvertedResidual(filters=256, strides=1, kernel_size=KERNEL_SIZE)(x)
    x = InvertedResidual(filters=256, strides=1, kernel_size=KERNEL_SIZE)(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = InvertedResidual(filters=512, strides=1, kernel_size=KERNEL_SIZE)(x)
    x = InvertedResidual(filters=512, strides=1, kernel_size=KERNEL_SIZE)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = DenseMoE(512, 2, expert_activation='relu', gating_activation='sigmoid')(x)
    o = layers.Dense(N_CLASSES, activity_regularizer=l2(
        LL2_REG), activation="sigmoid")(x)
    # delete above

    model = Model(inputs=i, outputs=o, name="conv2d")
    opt = tf.keras.optimizers.Adam(
        lr=LR, beta_1=0.9, beta_2=0.999, epsilon=EPSILON, decay=WEIGHT_DECAY, amsgrad=False
    )
    # class_acc = class_accuracy()
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING)
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=[
            calc_accuracy,
        ],
    )
    return model
    

def train_model(train_file, **args):
    logs_path = job_dir + "/logs/"
    print("-----------------------")
    print("Using module located at {}".format(__file__[-30:]))
    print("Using train_file located at {}".format(train_file))
    print("Using logs_path located at {}".format(logs_path))
    print("-----------------------")

    # setting variables
    print("Collecting Variables...")

    n_classes = params["N_CLASSES"]

    n_epochs = params["N_EPOCHS"]

    sr = params["SR"]
    lr = params["LR"]
    batch_size = params["BATCH_SIZE"]

    ll2_reg = params["LL2_REG"]
    weight_decay = params["WEIGHT_DECAY"]
    label_smoothing = params["LABEL_SMOOTHING"]

    epsilon = params["EPSILON"]

    shape = tuple(params["SHAPE"])

    es_patience = params["ES_PATIENCE"]
    min_delta = params["MIN_DELTA"]

    initial_channels = params["INITIAL_CHANNELS"]
    shape = shape + (initial_channels, )
    
    model_params = {
        "N_CLASSES": n_classes,
        "SR": sr,
        "BATCH_SIZE": batch_size,
        "LR": lr,
        "SHAPE": shape,
        "WEIGHT_DECAY": weight_decay,
        "LL2_REG": ll2_reg,
        "EPSILON": epsilon,
        "LABEL_SMOOTHING": label_smoothing
    }

    factor = params["FACTOR"]
    patience = params["PATIENCE"]
    min_lr = params["MIN_LR"]

    lr_params = {
        "factor": factor,
        "patience": patience,
        "min_lr": min_lr
    }

    print("Model Parameters: {}".format(model_params))
    print("Learning Rate Parameters: {}".format(lr_params))
    print("Early Stopping Patience and Delta: {}, {}%".format(es_patience, min_delta*100))
    print("-----------------------")

    train_test_ratio = 0.8

    filenames = []
    labels = []

    with tf.device('/CPU:0'): 

        train_file_name = train_file.split('/')[-1].split('.')[0]

        path = '../../data/txt_datasets/{}'.format(train_file_name)
        _list = os.path.join(path, 'aa_paths_and_labels.txt')
        with open(_list) as infile:
            for line in infile:
                elements = line.rstrip("\n").split(',')
                # elements[0] = '../' + elements[0]
                filenames.append(elements[0])
                labels.append((float(elements[1]), float(elements[2])))

        samples = list(zip(filenames, labels))

        random.shuffle(samples)

        val_samples, train_samples = train_test_split(
            samples, test_size=train_test_ratio)

        print("Size of training set: {}".format(len(train_samples)))
        print("Size of validation set: {}".format(len(val_samples)))

        train_filenames, train_labels = zip(*train_samples)
        val_filenames, val_labels = zip(*val_samples)

        train_dataset = tf.data.Dataset.from_tensor_slices((list(train_filenames), list(train_labels)))
        train_dataset = train_dataset.shuffle(len(train_filenames))
        # train_dataset = train_dataset.map(parse_function, num_parallel_calls=4)
        train_dataset = train_dataset.map(lambda filename, label: parse_function(filename, label, shape), num_parallel_calls=4)
        train_dataset = train_dataset.batch(batch_size)
        # print(batch_size)
        train_dataset = train_dataset.prefetch(1)

        val_dataset = tf.data.Dataset.from_tensor_slices((list(val_filenames), list(val_labels)))
        val_dataset = val_dataset.shuffle(len(val_filenames))
        val_dataset = val_dataset.map(lambda filename, label: parse_function(filename, label, shape), num_parallel_calls=4)
        # val_dataset = val_dataset.map(parse_function, num_parallel_calls=4)
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.prefetch(1)

        # weights
        weights = None

        if bool(params["CLASS_WEIGHTS"]):
            print("Initializing weights...")
            y_train = []
            for label in labels:
                y_train.append(convert_single(label))
            weights = class_weight.compute_class_weight(
                "balanced", np.unique(y_train), y_train)

            weights = {i: weights[i] for i in range(0, len(weights))}
            print("weights = {}".format(weights))
        
        # callbacks
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=1)
        tensorboard_callback = lr_tensorboard(log_dir=logs_path, histogram_freq=1)
        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_calc_accuracy", verbose=1, **lr_params
        )
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_calc_accuracy', min_delta=min_delta, patience=es_patience, verbose=1,
        )

    gpus = tf.config.experimental.list_logical_devices('GPU')

    # model setting
    model = conv2d(**model_params)

    model.summary()

    # if gpus:
    #     for gpu in gpus:
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=n_epochs,
        verbose=2,
        class_weight=weights,
        callbacks=[tensorboard_callback, reduce_lr_callback, early_stopping_callback]
    )
    


    model.save(job_dir + "/model.h5")

if __name__ == "__main__":
    print("Tensorflow Version: {}".format(tf.__version__))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    # tf.debugging.set_log_device_placement(True)
    # tf.debugging.set_log_device_placement(True)
    seed_everything()
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
        "--train-file",
        help="GCS or local paths to training data",
        required=True
    )

    parser.add_argument(
        "--job-dir",
        help="GCS location to write checkpoints and export models",
        required=True,
    )
    parser.add_argument(
        "--params",
        help="parameters used in the model and training",
        required=True,
    )
    args = parser.parse_args()
    arguments = args.__dict__
    job_dir = arguments.pop("job_dir")
    params = arguments.pop("params")
    params = json.loads(params)
    train_model(**arguments)
