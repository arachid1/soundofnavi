from .modules.helpers import *
from .modules.generators import *
from .modules.metrics import *
from .modules.callbacks import *
from .core import *

import argparse
import pickle
import tensorflow as tf
import sys
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
from sklearn.utils import class_weight
from tensorflow.python.lib.io import file_io
import json
import time
import numpy as np


def parse_function(filename, label, shape):

    spectrogram = tf.io.read_file(filename)
    # arr2 = tf.strings.split(arr, sep=',')
    spectrogram = tf.strings.split(spectrogram)
    # arr3 = tf.strings.unicode_decode(arr3, 'UTF-8')
    # print(arr2[:128])
    # print(arr3[0])
    spectrogram = tf.strings.split(spectrogram, sep=',')
    # print(tf.size(arr3))
    # print(type(arr3))
    spectrogram =tf.strings.to_number(spectrogram)
    spectrogram = tf.reshape(spectrogram.to_tensor(), (shape[0], shape[1]))
    spectrogram = tf.math.pow(spectrogram, 1/3)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    spectrogram = tf.tile(spectrogram, [1, 1, 3])
    return spectrogram, label

def teacher_(N_CLASSES, SR, BATCH_SIZE, LR, SHAPE, WEIGHT_DECAY, LL2_REG, EPSILON, LABEL_SMOOTHING):

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
    o = layers.Dense(N_CLASSES, activity_regularizer=l2(
        LL2_REG), activation="sigmoid")(x)
    # delete above

    model = Model(inputs=i, outputs=o, name="teacher")
    opt = tf.keras.optimizers.Adam(
        lr=LR, beta_1=0.9, beta_2=0.999, epsilon=EPSILON, decay=WEIGHT_DECAY, amsgrad=False
    )
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING)
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=[
            calc_accuracy,
        ],
    )
    return model

def student_(N_CLASSES, SR, BATCH_SIZE, LR, SHAPE, WEIGHT_DECAY, LL2_REG, EPSILON, LABEL_SMOOTHING):

    POOL_SIZE = (2, 2)
    i = layers.Input(shape=SHAPE)
    x = layers.BatchNormalization()(i)
    tower_1 = layers.Conv2D(8, (1,1), padding='same', activation='relu')(x)
    tower_1 = layers.Conv2D(8, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = layers.Conv2D(8, (1,1), padding='same', activation='relu')(x)
    tower_2 = layers.Conv2D(8, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    tower_3 = layers.Conv2D(8, (1,1), padding='same', activation='relu')(tower_3)
    x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    tower_1 = layers.Conv2D(16, (1,1), padding='same', activation='relu')(x)
    tower_1 = layers.Conv2D(16, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = layers.Conv2D(16, (1,1), padding='same', activation='relu')(x)
    tower_2 = layers.Conv2D(16, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    tower_3 = layers.Conv2D(16, (1,1), padding='same', activation='relu')(tower_3)
    x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    tower_1 = layers.Conv2D(32, (1,1), padding='same', activation='relu')(x)
    tower_1 = layers.Conv2D(32, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = layers.Conv2D(32, (1,1), padding='same', activation='relu')(x)
    tower_2 = layers.Conv2D(32, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    tower_3 = layers.Conv2D(32, (1,1), padding='same', activation='relu')(tower_3)
    x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    tower_1 = layers.Conv2D(64, (1,1), padding='same', activation='relu')(x)
    tower_1 = layers.Conv2D(64, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = layers.Conv2D(64, (1,1), padding='same', activation='relu')(x)
    tower_2 = layers.Conv2D(64, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    tower_3 = layers.Conv2D(64, (1,1), padding='same', activation='relu')(tower_3)
    x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    o = layers.Dense(N_CLASSES, activity_regularizer=l2(
        LL2_REG), activation="sigmoid")(x)

    model = Model(inputs=i, outputs=o, name="student")
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

    # Initializing models
    
    file_path = "../../cache/conv___03_1714___all_sw_coch_preprocessed_v2_param_v21_augm_v0_cleaned_8000___mod9_64ch_cuberooted_normal_bsize32___2012/model.h5"
    teacher = teacher_(**model_params)
    teacher.load_weights(file_path)
    teacher.summary()

    student = student_(**model_params)
    student.summary()

    # student_scratch = tf.keras.models.clone_model(student)

    # Compile and training distiller
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
    opt = tf.keras.optimizers.Adam(
        lr=lr, beta_1=0.9, beta_2=0.999, epsilon=epsilon, decay=weight_decay, amsgrad=False
    )

    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=opt,
        metrics=[calc_accuracy],
        student_loss_fn=loss,
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=10,
    )

    if gpus:
        for gpu in gpus:
            distiller.fit(train_dataset,
            validation_data=val_dataset,
            verbose=2,
            epochs=n_epochs,
            # class_weight=weights,
            callbacks=[tensorboard_callback, reduce_lr_callback, early_stopping_callback])

    # Compile and train student clone
    # print("Training student model alone.")

    # student_scratch.compile(
    #     optimizer=opt,
    #     loss=loss,
    #     metrics=[
    #         calc_accuracy,
    #     ],
    # )
    # if gpus:
    #     for gpu in gpus:
    #         student_scratch.fit(train_dataset, 
    #         validation_data=val_dataset, 
    #         verbose=2, 
    #         epochs=n_epochs,
    #         # class_weight=weights,
    #         callbacks=[tensorboard_callback, reduce_lr_callback, early_stopping_callback])

    teacher_to_save, student_to_save = distiller.return_models()

    teacher_to_save.save(job_dir + "/teacher.h5")
    student_to_save.save(job_dir + "/student.h5")

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
