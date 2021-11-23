from .modules.helpers import *
from .modules.generators import *
from .modules.metrics import *
from .modules.callbacks import *
from .modules.pneumonia import *
from .modules.parse_functions import *
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
from sklearn.model_selection import StratifiedKFold, KFold
import json
import time
import numpy as np
import glob
import wandb
from wandb.keras import WandbCallback

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

    six = bool(params["SIX"])
    concat = bool(params["CONCAT"])

    epoch_start = int(params["EPOCH_START"])
    target = int(params["TARGET"])

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
    print("Six: {} and Concat: {}".format(six, concat))
    print("-----------------------")

    train_test_ratio = 0.8

    filenames = []
    labels = []

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

    filenames, labels = process_bangladesh(six, dataset_path)

    samples = list(zip(filenames, labels))

    random.shuffle(samples)

    # print(samples[0])
    # print(samples[1])

    filenames, labels = zip(*samples)
    # print(filenames[0])
    # print(labels[0])
    # print(filenames[1])
    # print(labels[1])

    kf = StratifiedKFold(n_splits=10)

    split_count = 0

    for train_indexes, val_indexes in kf.split(filenames, labels):
        print("K-Fold split N: {}".format(split_count))

        run = wandb.init(project="kfold_crossvalidation_model9", mode="disabled", sync_tensorboard=False, reinit=True)
        # test_index = val_indexes[0]
        # print(test_index)
        # print(filenames[test_index])
        # print(labels[test_index])
        # exit()
        grouped_val_samples = [(filenames[i], labels[i]) for i in val_indexes]
        grouped_train_samples = [(filenames[i], labels[i]) for i in train_indexes]

        random.shuffle(grouped_val_samples)
        random.shuffle(grouped_train_samples)

        if concat:
            # if not six:
            #     print("You either (1) have set concatenate to true and  didn't set SIX to true. ")
            #     print("or (2) you set initial_channels to {} instead of 6".format(initial_channels))
            #     exit()
            val_dataset, train_dataset = process_data(grouped_val_samples, grouped_train_samples, concatenate_specs, shape, batch_size, initial_channels)
        else:
            train_samples = [(recording, s[1]) for s in grouped_train_samples for recording in s[0]]
            # print(grouped_val_samples)
            # print(len(grouped_val_samples))
            val_samples = [(recording, s[1]) for s in grouped_val_samples for recording in s[0]]
            # print(len(val_samples))
            # print(len(val_samples))
            print("With concat = {} and six = {}, size of training set: {}...".format(concat, six, len(train_samples)))
            print("...and size of validation set: {}".format(len(val_samples)))
            val_dataset, train_dataset = process_data(val_samples, train_samples, generate_spec, shape, batch_size, initial_channels)
        
        # weights
        weights = None

        if bool(params["CLASS_WEIGHTS"]):
            print("Initializing weights...")
            weights = class_weight.compute_class_weight(
                "balanced", np.unique(labels), labels)
            weights = {i: weights[i] for i in range(0, len(weights))}
            print("weights = {}".format(weights))
        
        # callbacks
        metrics_callback = metric_callback(val_dataset, shape, n_classes, job_dir, min_delta, es_patience, patience, min_lr, factor, epoch_start, target, split_count)

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
                excel_dest = job_dir + "/validation_sheet_{}_{}.xls".format(split_count, epoch)
                bd_f1, bd_accuracy, bd_precision, bd_recall, bd_auc, bd_y_true, bd_y_pred = generate_bangladesh_sheet(model, grouped_val_samples, excel_dest, six, initial_channels, pneumonia_threshold)
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
        
        split_count += 1
        run.finish()


if __name__ == "__main__":
    print("Tensorflow Version: {}".format(tf.__version__))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
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
