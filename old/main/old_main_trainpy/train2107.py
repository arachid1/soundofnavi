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
    # class_acc = class_accuracy()
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

    wandb.init(project="tensorboard-integration", sync_tensorboard=False)

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

    with tf.device('/CPU:0'): 

        train_file_name = train_file.split('/')[-1].split('.')[0]

        dataset_path = '../../data/txt_datasets/{}'.format(train_file_name)
        # _list = os.path.join(path, 'aa_paths_and_labels.txt')
        # with open(_list) as infile:
        #     for line in infile:
        #         elements = line.rstrip("\n").split(',')
        #         # elements[0] = '../' + elements[0]
        #         filenames.append(elements[0])
        #         labels.append((float(elements[1]))

        # print("here")
        filenames, labels = process_bangladesh(six, dataset_path)

        # print(len(filenames))

        samples = list(zip(filenames, labels))

        # samples = samples[:20]

        random.shuffle(samples)

        grouped_val_samples, grouped_train_samples = train_test_split(
                samples, test_size=train_test_ratio)

        random.shuffle(grouped_val_samples)
        random.shuffle(grouped_train_samples)

        # print(grouped_val_samples[0])
        # exit()

        # print(len(train_samples))
        # print(len(val_samples))
        # print(val_samples[1])
        # print(train_samples[0])
        # print(train_samples[1])
        
        if concat:
            if not six or not (initial_channels == 6):
                print("You either (1) have set concatenate to true and  didn't set SIX to true. ")
                print("or (2) you set initial_channels to {} instead of 6".format(initial_channels))
                exit()
            val_dataset, train_dataset = process_data(grouped_val_samples, grouped_train_samples, concatenate_specs, shape, batch_size, initial_channels)
        else:
            print("here")
            # print(train_samples[0])
            train_samples = [(recording, s[1]) for s in grouped_train_samples for recording in s[0]]
            # print(grouped_val_samples)
            # print(len(grouped_val_samples))
            val_samples = [(recording, s[1]) for s in grouped_val_samples for recording in s[0]]
            # print(len(val_samples))
            # print(len(val_samples))
            # exit()
            # print(train_samples[:7])
            print("With concat = {} and six = {}, size of training set: {}...".format(concat, six, len(train_samples)))
            print("...and size of validation set: {}".format(len(val_samples)))
            # exit()
            val_dataset, train_dataset = process_data(val_samples, train_samples, generate_spec, shape, batch_size, initial_channels)
        
        # print(len(train_samples))
        # print(len(val_samples))
        


        # val_samples, train_samples = train_test_split(
        #     samples, test_size=train_test_ratio)

        # print("Size of training set: {}".format(len(train_samples)))
        # print("Size of validation set: {}".format(len(val_samples)))

        # train_filenames, train_labels = zip(*train_samples)
        # val_filenames, val_labels = zip(*val_samples)

        # train_dataset = tf.data.Dataset.from_tensor_slices((list(train_filenames), list(train_labels)))
        # train_dataset = train_dataset.shuffle(len(train_filenames))
        # # train_dataset = train_dataset.map(parse_function, num_parallel_calls=4)
        # train_dataset = train_dataset.map(lambda filename, label: parse_function(filename, label, shape), num_parallel_calls=4)
        # train_dataset = train_dataset.batch(batch_size)
        # # print(batch_size)
        # train_dataset = train_dataset.prefetch(1)

        # val_dataset = tf.data.Dataset.from_tensor_slices((list(val_filenames), list(val_labels)))
        # val_dataset = val_dataset.shuffle(len(val_filenames))
        # val_dataset = val_dataset.map(lambda filename, label: parse_function(filename, label, shape), num_parallel_calls=4)
        # # val_dataset = val_dataset.map(parse_function, num_parallel_calls=4)
        # val_dataset = val_dataset.batch(batch_size)
        # val_dataset = val_dataset.prefetch(1)

        # weights
        weights = None

        if bool(params["CLASS_WEIGHTS"]):
            print("Initializing weights...")
            # y_train = []
            # for label in labels:
            #     y_train.append(convert_single(label))
            weights = class_weight.compute_class_weight(
                "balanced", np.unique(labels), labels)

            weights = {i: weights[i] for i in range(0, len(weights))}
            print("weights = {}".format(weights))
        
        # callbacks
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=1)
        metrics_callback = metric_callback(val_dataset, shape, n_classes, job_dir, min_delta, es_patience, patience, min_lr, factor)
        # tensorboard_callback = lr_tensorboard(log_dir=wandb.run.dir, histogram_freq=1)
        # reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        #     monitor="val_accuracy", verbose=1, **lr_params
        # )
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', min_delta=min_delta, patience=es_patience, verbose=1,
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
                callbacks=[metrics_callback]
            )

    model.save(job_dir + "/model_{}.h5".format(n_epochs))

    

    print("\nStarting the Bangladesh analysis...")
    models = glob.glob(os.path.join(job_dir, "*.h5"))
    best_bd_f1 = 0
    for model_path in models:
        # print(model_path)
        model = conv2d(**model_params)
        model.load_weights(model_path)
        # print(model)
        epoch = model_path.split('/')[-1].split('.')[0].split('_')[-1]
        print("Epoch: {}".format(epoch))
        excel_dest = job_dir + "/validation_sheet_{}.xls".format(epoch)
        bd_f1, bd_accuracy, bd_precision, bd_recall = generate_bangladesh_sheet(model, grouped_val_samples, excel_dest, six, initial_channels)
        if bd_f1 > best_bd_f1:
            best_bd_f1 = bd_f1
            wandb.run.summary.update({"best_bd_f1": bd_f1})
            wandb.run.summary.update({"best_bd_accuracy": bd_accuracy})
            wandb.run.summary.update({"best_bd_precision": bd_precision})
            wandb.run.summary.update({"best_bd_recall": bd_recall})
            wandb.run.summary.update({"best_bd_epoch": epoch})
            # wandb.run.summary.update({"best_val_accuracy": self.best_accuracy})
            # wandb.run.summary.update({"best_val_precision": self.best_precision})
            # wandb.run.summary.update({"best_val_recall": self.best_recall})
            # wandb.run.summary.update({"best_auc": self.best_auc})
            # wandb.run.summary.update({"best_epoch": self.best_epoch})

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
