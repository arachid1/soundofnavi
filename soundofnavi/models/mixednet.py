import pandas as pd
import numpy as np
import tensorflow as tf
from .core import *
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from kapre.composed import get_melspectrogram_layer
from kapre.time_frequency import STFT, Magnitude, Phase, MagnitudeToDecibel, ApplyFilterbank
from ..main import parameters
    
def mixednet(N_CLASSES, SR, BATCH_SIZE, LR, SHAPE, INITIAL_CHANNELS, WEIGHT_DECAY, LL2_REG, EPSILON, LABEL_SMOOTHING):

    KERNELS = [3, 4, 5, 6, 7]
    POOL_SIZE=2
    i = layers.Input(shape=SHAPE + (INITIAL_CHANNELS,))
    x = layers.Conv2D(8, (3, 3), padding='same', activation='relu')(i)
    x = MixDepthGroupConvolution2D(kernels=KERNELS)(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = MixDepthGroupConvolution2D(kernels=KERNELS)(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MixDepthGroupConvolution2D(kernels=KERNELS)(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MixDepthGroupConvolution2D(kernels=KERNELS)(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MixDepthGroupConvolution2D(kernels=KERNELS)(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    o = layers.Dense(N_CLASSES, activity_regularizer=l2(
        LL2_REG), activation="sigmoid")(x)
        # delete above

    model = Model(inputs=i, outputs=o, name="mixednet")
    opt = tf.keras.optimizers.Adam(
        lr=LR, beta_1=0.9, beta_2=0.999, epsilon=EPSILON, decay=WEIGHT_DECAY, amsgrad=False
    )
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING)
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=[
            'accuracy', 
        ],
    )
    return model