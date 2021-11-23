from .modules.helpers import *
from .modules.generators import *
from .modules.metrics import *
from .modules.callbacks import *
from .modules.pneumonia import *
from .modules.parse_functions import *
from .modules.augmentation import *
from .core import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import argparse
import pickle
import tensorflow as tf
import tensorflow_addons
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
import glob
import wandb
from wandb.keras import WandbCallback

def conv2d(N_CLASSES, SR, BATCH_SIZE, LR, SHAPE, WEIGHT_DECAY, LL2_REG, EPSILON, LABEL_SMOOTHING):

    KERNELS = [3, 4, 5, 6, 7]
    KERNEL_SIZE = 6
    POOL_SIZE = (2, 2)
    i = layers.Input(shape=SHAPE, )
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
    # x = TimeDistributedInvertedResidual(filters=24, strides=1, expansion_factor=3, kernel_size=KERNEL_SIZE)(x)
    # x = TimeDistributedInvertedResidual(filters=24, strides=1, expansion_factor=3, kernel_size=KERNEL_SIZE)(x)
    # x = layers.TimeDistributed(layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same"))(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.TimeDistributed(MixDepthGroupConvolution2D(kernels=KERNELS))(x)
    x = layers.TimeDistributed(MixDepthGroupConvolution2D(kernels=KERNELS))(x)
    # x = layers.TimeDistributed(layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same"))(x)
    # x = layers.BatchNormalization()(x)
    # x = TimeDistributedInvertedResidual(filters=64, strides=1, kernel_size=KERNEL_SIZE)(x)
    # x = TimeDistributedInvertedResidual(filters=64, strides=1, kernel_size=KERNEL_SIZE)(x)
    # x = layers.TimeDistributed(layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same"))(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dense(N_CLASSES, activity_regularizer=l2(
        LL2_REG), activation="sigmoid")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64)(x)
    o = layers.Dense(N_CLASSES, activity_regularizer=l2(
        LL2_REG), activation="sigmoid")(x)
    # delete above

    model = Model(inputs=i, outputs=o, name="conv2d")
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
    

def train_model(train_file, job_dir, params, wav_params, spec_params):
    logs_path = job_dir + "/logs/"

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

    job_version = params["PARAM"]

    es_patience = params["ES_PATIENCE"]
    min_delta = params["MIN_DELTA"]

    six = bool(params["SIX"])
    concat = bool(params["CONCAT"])

    epoch_start = int(params["EPOCH_START"])
    target = int(params["TARGET"])
    clause = int(params["CLAUSE"])
    testing = int(params["TESTING"])
    cuberooting = int(params["CUBEROOTING"])

    initial_channels = params["INITIAL_CHANNELS"]
    # shape = shape + (initial_channels, )
    shape = (5, 250, 128, 1)
    
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

    augmentation = bool(params['AUGMENTATION'])
    spec_add = bool(spec_params['ADD'])
    spec_quantity = int(spec_params["QUANTITY"])
    time_masking = int(spec_params["TIME_MASKING"])
    frequency_masking = int(spec_params["FREQUENCY_MASKING"])
    loudness = int(spec_params["LOUDNESS"])

    wav_add = bool(wav_params['ADD'])
    wav_quantity = int(wav_params['QUANTITY'])
    wav_path = str(wav_params["WAV_PATH"]).split(',')

    print("All variables have been collected.")

    train_test_ratio = 0.8

    split_count = 0
    filenames = []
    labels = []

    wandb_name = __file__.split('/')[-1].split('.')[0]
    # print(wandb_name)
    wandb.init(project="tensorboard-integration", name=wandb_name, sync_tensorboard=False)

    config = wandb.config

    config.n_classes = n_classes
    config.n_epochs = n_epochs
    config.sr = sr
    config.lr = lr
    config.batch_size = batch_size
    config.ll2_reg = ll2_reg
    config.weight_decay = weight_decay
    config.label_smoothing = label_smoothing
    config.es_patience = es_patience
    config.min_delta = min_delta
    config.initial_channels = initial_channels
    config.factor = factor
    config.patience = patience
    config.min_lr = min_lr

    train_file_name = train_file.split('/')[-1].split('.')[0]

    dataset_path = '../../data/txt_datasets/{}'.format(train_file_name)
    # augmentation_path = dataset_path.split('/')

    filenames, labels = process_bangladesh(six, dataset_path)

    nb_pneumonia = labels.count(1)
    nb_non_pneumonia = labels.count(0)
    # print("There are {} pneumonia patients and {} non-pneumonia patients".format(nb_pneumonia, nb_non_pneumonia))
    print("Pneumonia patients: {}, Non-pneumonia patients: {}".format(nb_pneumonia, nb_non_pneumonia))

    samples = list(zip(filenames, labels))

    if testing:
        samples = samples[:20]
        n_epochs = 10
        spec_quantity = 10
        wav_quantity = 10

    random.shuffle(samples)

    __, stratify_labels = zip(*samples)

    grouped_val_samples, grouped_train_samples = train_test_split(
            samples, test_size=train_test_ratio, stratify=stratify_labels)

    random.shuffle(grouped_val_samples)
    random.shuffle(grouped_train_samples)

    if concat:
        val_dataset, train_dataset = process_data(grouped_val_samples, grouped_train_samples, concatenate_specs, shape, batch_size, initial_channels, cuberooting)
    else:
        print("-----------------------")
        train_samples = [[recording, s[1]] for s in grouped_train_samples for recording in s[0]]
        original_length = len(train_samples)
        train_labels = [label for _, label in train_samples] 
        nb_train_pneumonia, nb_train_non_pneumonia = train_labels.count(1), train_labels.count(0)
        print("Number of train recordings: {} with pneumonia: {} and non-pneumonia: {}".format(original_length, nb_train_pneumonia, nb_train_non_pneumonia))

        if augmentation:
            train_samples, train_labels = manage_augmentations(train_samples, wav_add, spec_add, wav_quantity, spec_quantity, job_version, wav_path, time_masking, frequency_masking, loudness, dataset_path)

        val_samples = [[recording, s[1]] for s in grouped_val_samples for recording in s[0]]
        val_labels = [label for _, label in val_samples] 
        nb_val_pneumonia, nb_val_non_pneumonia = val_labels.count(1), val_labels.count(0)
        print("Number of val recordings: {} with pneumonia: {} and non-pneumonia: {}".format(len(val_samples), nb_val_pneumonia, nb_val_non_pneumonia))
        print("-----------------------")

        lr = 1e-3 * (original_length / len(train_samples))

        print("The initial learning rate for this job is: {}".format(lr))

        # print("With concat = {} and six = {}, size of training set: {}...".format(concat, six, len(train_samples)))
        # print("...and size of validation set: {}".format(len(val_samples)))
        val_dataset, train_dataset = process_data(val_samples, train_samples, generate_timed_spec, shape, batch_size, initial_channels, cuberooting)
    
    # weights
    weights = None

    if bool(params["CLASS_WEIGHTS"]):
        print("Initializing weights...")
        weights = class_weight.compute_class_weight(
            "balanced", np.unique(train_labels), train_labels)
        weights = {i: weights[i] for i in range(0, len(weights))}
        print("weights = {}".format(weights))
    
    # callbacks
    metrics_callback = metric_callback(val_dataset, shape, n_classes, job_dir, min_delta, es_patience, patience, min_lr, factor, epoch_start, target, split_count, clause)

    gpus = tf.config.experimental.list_logical_devices('GPU')

    # model setting
    model = conv2d(**model_params)

    model.summary(line_length=110)

    if gpus:
        if len(gpus) > 1:
            print("You are using 2 GPUs while the code is set up for one only.")
            exit()
        for gpu in gpus:
            model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=n_epochs,
                verbose=2,
                class_weight=weights,
                callbacks=[metrics_callback]
            )

    model.save(job_dir + "/{}/model_{}.h5".format(split_count, n_epochs))

    print("\nStarting the Bangladesh analysis...")
    models = sorted(glob.glob(os.path.join(job_dir, "{}/*.h5".format(split_count))))
    best_bd_target_metric = 0
    thresholds = [1, 2, 3]
    for model_path in models:
        for pneumonia_threshold in thresholds:
            # print(model_path)
            model = conv2d(**model_params)
            model.load_weights(model_path)
            # print(model)
            epoch = model_path.split('/')[-1].split('.')[0].split('_')[-1]
            print("Epoch: {}".format(epoch))
            excel_dest = job_dir + "/validation_sheet_{}.xls".format(epoch)
            bd_f1, bd_accuracy, bd_precision, bd_recall, bd_auc, bd_y_true, bd_y_pred = generate_bangladesh_sheet(model, shape, grouped_val_samples, excel_dest, six, initial_channels, pneumonia_threshold)
            if target == 0: # auc 
                target_metric = bd_auc
            elif target == 1: # f1
                target_metric = bd_f1
            if target_metric > best_bd_target_metric:
                best_bd_target_metric = target_metric
                wandb.run.summary.update({"best_bd_f1": bd_f1})
                wandb.run.summary.update({"best_bd_auc": bd_auc})
                wandb.run.summary.update({"best_bd_accuracy": bd_accuracy})
                wandb.run.summary.update({"best_bd_precision": bd_precision})
                wandb.run.summary.update({"best_bd_recall": bd_recall})
                wandb.run.summary.update({"best_bd_epoch": epoch})
                wandb.log({"bd_conf_mat" : wandb.plot.confusion_matrix(probs=None,
                                y_true=bd_y_true, preds=bd_y_pred,) })
            if testing:
                exit()
                # wandb.run.summary.update({"best_val_accuracy": self.best_accuracy})
                # wandb.run.summary.update({"best_val_precision": self.best_precision})
                # wandb.run.summary.update({"best_val_recall": self.best_recall})
                # wandb.run.summary.update({"best_auc": self.best_auc})
                # wandb.run.summary.update({"best_epoch": self.best_epoch})

if __name__ == "__main__":
    print("Tensorflow Version: {}".format(tf.__version__))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
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
    parser.add_argument(
        "--wav-params",
        help="Augmentation parameters for audio files",
        required=True,
    )
    parser.add_argument(
        "--spec-params",
        help="Augmentation parameters for spectrogram",
        required=True,
    )
    args = parser.parse_args()
    arguments = args.__dict__
    train_file = arguments.pop("train_file")
    job_dir = arguments.pop("job_dir")
    params = arguments.pop("params")
    params = json.loads(params)
    wav_params = arguments.pop("wav_params")
    wav_params = json.loads(wav_params)
    spec_params = arguments.pop("spec_params")
    spec_params = json.loads(spec_params)
    print("-----------------------")
    print("Using module: {}".format(__file__[-15:]))
    print("Using train_file: {}".format(train_file))
    print("Job directory: {}".format(job_dir))
    print("Parameters: {}".format(params))
    print("Augmentation parameters for audio: {}".format(wav_params))
    print("Augmentation parameters for spectrograms: {}".format(spec_params))
    print("-----------------------")
    train_model(train_file, job_dir, params, wav_params, spec_params)