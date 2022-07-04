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
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
from sklearn.utils import class_weight
from tensorflow.python.lib.io import file_io
import json
import time
import numpy as np

def conv2d(N_CLASSES, SR, BATCH_SIZE, LR, SHAPE, WEIGHT_DECAY, LL2_REG, EPSILON, LABEL_SMOOTHING):

    KERNEL_SIZE = (3, 3)
    POOL_SIZE = (2, 2)
    PADDING = "same"
    DROPOUT = 0.1
    i = layers.Input(shape=SHAPE,)
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
    tower_1 = layers.Conv2D(24, (1,1), padding='same', activation='relu')(x)
    tower_1 = layers.Conv2D(24, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = layers.Conv2D(24, (1,1), padding='same', activation='relu')(x)
    tower_2 = layers.Conv2D(24, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    tower_3 = layers.Conv2D(24, (1,1), padding='same', activation='relu')(tower_3)
    x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    tower_1 = layers.Conv2D(72, (1,1), padding='same', activation='relu')(x)
    tower_1 = layers.Conv2D(72, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = layers.Conv2D(72, (1,1), padding='same', activation='relu')(x)
    tower_2 = layers.Conv2D(72, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    tower_3 = layers.Conv2D(72, (1,1), padding='same', activation='relu')(tower_3)
    x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    tower_1 = layers.Conv2D(216, (1,1), padding='same', activation='relu')(x)
    tower_1 = layers.Conv2D(216, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = layers.Conv2D(216, (1,1), padding='same', activation='relu')(x)
    tower_2 = layers.Conv2D(216, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    tower_3 = layers.Conv2D(216, (1,1), padding='same', activation='relu')(tower_3)
    x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    # tower_1 = layers.Conv2D(128, (1,1), padding='same', activation='relu')(x)
    # tower_1 = layers.Conv2D(128, (3,3), padding='same', activation='relu')(tower_1)
    # tower_2 = layers.Conv2D(128, (1,1), padding='same', activation='relu')(x)
    # tower_2 = layers.Conv2D(128, (5,5), padding='same', activation='relu')(tower_2)
    # tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    # tower_3 = layers.Conv2D(128, (1,1), padding='same', activation='relu')(tower_3)
    # x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    # x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    # x = layers.BatchNormalization()(x)
    # tower_1 = layers.Conv2D(256, (1,1), padding='same', activation='relu')(x)
    # tower_1 = layers.Conv2D(256, (3,3), padding='same', activation='relu')(tower_1)
    # tower_2 = layers.Conv2D(256, (1,1), padding='same', activation='relu')(x)
    # tower_2 = layers.Conv2D(256, (5,5), padding='same', activation='relu')(tower_2)
    # tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    # tower_3 = layers.Conv2D(256, (1,1), padding='same', activation='relu')(tower_3)
    # x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    # x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    # x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
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

    if gpus:
        for gpu in gpus:
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
