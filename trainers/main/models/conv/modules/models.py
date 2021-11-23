import pandas as pd
import numpy as np
import tensorflow as tf
from .core import *
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from kapre.composed import get_melspectrogram_layer
from kapre.time_frequency import STFT, Magnitude, Phase, MagnitudeToDecibel, ApplyFilterbank
from .main import parameters
    
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

def model9(N_CLASSES, SR, BATCH_SIZE, LR, SHAPE, INITIAL_CHANNELS, WEIGHT_DECAY, LL2_REG, EPSILON, LABEL_SMOOTHING):

    KERNEL_SIZE = 6
    POOL_SIZE = (2, 2)
    i = layers.Input(shape=SHAPE + (INITIAL_CHANNELS,))
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
    o = layers.Dense(N_CLASSES, activity_regularizer=l2(
        LL2_REG), activation="sigmoid")(x)
    # delete above

    model = Model(inputs=i, outputs=o, name="model9")
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

def custom_mixednet(N_CLASSES, SR, BATCH_SIZE, LR, SHAPE, INITIAL_CHANNELS, WEIGHT_DECAY, LL2_REG, EPSILON, LABEL_SMOOTHING):

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

    # metrics
    roc_auc = tf.keras.metrics.AUC()

    model = Model(inputs=i, outputs=o, name="custom_mixednet")
    opt = tf.keras.optimizers.Adam(
        lr=LR, beta_1=0.9, beta_2=0.999, epsilon=EPSILON, decay=WEIGHT_DECAY, amsgrad=False
    )
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING)
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=[
            'accuracy',
            roc_auc 
        ],
    )
    return model

def time_series_model(N_CLASSES, SR, BATCH_SIZE, LR, SHAPE, INITIAL_CHANNELS, WEIGHT_DECAY, LL2_REG, EPSILON, LABEL_SMOOTHING):

    KERNEL_SIZE = 6
    POOL_SIZE = (2, 2)
    SHAPE = (parameters.n_sequences, int(SHAPE[1]/parameters.n_sequences), SHAPE[0])
    i = layers.Input(shape=SHAPE + (INITIAL_CHANNELS,))
    x = layers.BatchNormalization()(i)
    tower_1 = layers.TimeDistributed(layers.Conv2D(16, (1,1), padding='same', activation='relu'))(x)
    tower_1 = layers.TimeDistributed(layers.Conv2D(16, (3,3), padding='same', activation='relu'))(tower_1)
    tower_2 = layers.TimeDistributed(layers.Conv2D(16, (1,1), padding='same', activation='relu'))(x)
    tower_2 = layers.TimeDistributed(layers.Conv2D(16, (5,5), padding='same', activation='relu'))(tower_2)
    tower_3 = layers.TimeDistributed(layers.MaxPooling2D((3,3), strides=(1,1), padding='same'))(x)
    tower_3 = layers.TimeDistributed(layers.Conv2D(16, (1,1), padding='same', activation='relu'))(tower_3)
    x = layers.Concatenate(axis=-1)([tower_1, tower_2, tower_3])
    x = layers.TimeDistributed(layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same"))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    # x = TimeDistributedInvertedResidual(filters=64, strides=1, kernel_size=KERNEL_SIZE)(x)
    # x = TimeDistributedInvertedResidual(filters=64, strides=1, kernel_size=KERNEL_SIZE)(x)
    # x = layers.TimeDistributed(layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same"))(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dense(128)(x)
    x = layers.Dense(N_CLASSES, activity_regularizer=l2(
        LL2_REG), activation="sigmoid")(x)
    x = layers.Flatten()(x)
    o = layers.Dense(N_CLASSES, activity_regularizer=l2(
        LL2_REG), activation="sigmoid")(x)
    # delete above

    model = Model(inputs=i, outputs=o, name="time_series_model")
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

def kapre_model(N_CLASSES, SR, BATCH_SIZE, LR, SHAPE, INITIAL_CHANNELS, WEIGHT_DECAY, LL2_REG, EPSILON, LABEL_SMOOTHING):

    KERNEL_SIZE = 6
    KERNELS = [3, 4, 5, 6, 7]
    POOL_SIZE = (2, 2)
    waveform_to_stft = STFT(
        n_fft=parameters.n_fft,
        hop_length=parameters.hop_length,
    )
    stft_to_stftm = Magnitude()
    kwargs = {
        'sample_rate': parameters.sr,
        'n_freq': parameters.n_fft // 2 + 1,
        'n_mels': parameters.n_mels
    }
    stftm_to_melgram = ApplyFilterbank(
        type='mel', trainable=parameters.trainable_fb, filterbank_kwargs=kwargs, 
    )
    mag_to_decibel = MagnitudeToDecibel()
    i = layers.Input(shape=SHAPE + (1,))
    x = waveform_to_stft(i)
    x = stft_to_stftm(x)
    x = stftm_to_melgram(x)
    if parameters.to_decibel:
        x = mag_to_decibel(x)
    x = layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
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

    model = Model(inputs=i, outputs=o, name="audio_model")
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

def audiomod1(N_CLASSES, SR, BATCH_SIZE, LR, SHAPE, INITIAL_CHANNELS, WEIGHT_DECAY, LL2_REG, EPSILON, LABEL_SMOOTHING):

    waveform_to_stft = STFT(
        n_fft=parameters.n_fft,
        hop_length=parameters.hop_length,
    )
    stft_to_stftm = Magnitude()
    kwargs = {
        'sample_rate': parameters.sr,
        'n_freq': parameters.n_fft // 2 + 1,
        # 'n_bins': 80,
        # 'n_mels': parameters.n_mels,
    }
    stftm_to_melgram = ApplyFilterbank(
        type='mel', trainable=parameters.trainable_fb, filterbank_kwargs=kwargs, 
    )
    mag_to_decibel = MagnitudeToDecibel()
    KERNEL_SIZE = 6
    KERNELS = [3, 4, 5, 6, 7]
    POOL_SIZE = (2, 2)
    i = layers.Input(shape=SHAPE + (1,))
    x = waveform_to_stft(i)
    x = stft_to_stftm(x)
    # x = layers.Dropout(0.1)(x)
    x = stftm_to_melgram(x)
    if parameters.to_decibel:
        x = mag_to_decibel(x)
    # x =  layers.Dropout(0.1)(x)
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
    x = layers.GlobalAveragePooling2D()(x)
    o = layers.Dense(N_CLASSES, activity_regularizer=l2(
        LL2_REG), activation="sigmoid")(x)
    # delete above

    model = Model(inputs=i, outputs=o, name="audio_model")
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

def audiomod2(N_CLASSES, SR, BATCH_SIZE, LR, SHAPE, INITIAL_CHANNELS, WEIGHT_DECAY, LL2_REG, EPSILON, LABEL_SMOOTHING):

    waveform_to_stft = STFT(
        n_fft=parameters.n_fft,
        hop_length=parameters.hop_length,
    )
    stft_to_stftm = Magnitude()
    kwargs = {
        'sample_rate': parameters.sr,
        'n_freq': parameters.n_fft // 2 + 1,
        'n_mels': parameters.n_mels,
    }
    stftm_to_melgram = ApplyFilterbank(
        type='mel', trainable=parameters.trainable_fb, filterbank_kwargs=kwargs, 
    )
    mag_to_decibel = MagnitudeToDecibel()

    KERNEL_SIZE = 6
    KERNELS = [3, 4, 5, 6, 7]
    POOL_SIZE = (2, 2)

    i = layers.Input(shape=SHAPE + (1,))

    x = waveform_to_stft(i)
    x = stft_to_stftm(x)
    # x = stftm_to_melgram(x)
    if parameters.to_decibel:
        x = mag_to_decibel(x)
    
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
    x = layers.GlobalAveragePooling2D()(x)
    o = layers.Dense(N_CLASSES, activity_regularizer=l2(
        LL2_REG), activation="sigmoid")(x)
    # delete above

    model = Model(inputs=i, outputs=o, name="audio_model")
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