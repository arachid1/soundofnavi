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

def generate_spec_custom(filename, label, shape, initial_channels):

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
    # print(shape[0])
    # print(shape[1])
    # exit()
    spectrogram = tf.reshape(spectrogram.to_tensor(), (shape[0], shape[1]))
    # spectrogram = tf.expand_dims(spectrogram, axis=-1)
    # spectrogram = tf.tile(spectrogram, [1, 1, initial_channels])
    return spectrogram, label

def conv2d(N_CLASSES, SR, BATCH_SIZE, LR, SHAPE, WEIGHT_DECAY, LL2_REG, EPSILON, LABEL_SMOOTHING):

    # KERNELS = [3, 4, 5, 6, 7]
    # POOL_SIZE=2
    i = layers.Input(shape=SHAPE, )
    model_2_1 = layers.GRU(32,return_sequences=True,activation=None,go_backwards=True)(i)
    model_2 = layers.LeakyReLU()(model_2_1)
    model_2 = layers.GRU(128,return_sequences=True, activation=None,go_backwards=True)(model_2)
    #model_2 = BatchNormalization()(model_2)
    model_2 = layers.LeakyReLU()(model_2)
    
    model_3 = layers.GRU(64,return_sequences=True,activation=None,go_backwards=True)(i)
    model_3 = layers.LeakyReLU()(model_3)
    model_3 = layers.GRU(128,return_sequences=True, activation=None,go_backwards=True)(model_3)
    #model_3 = BatchNormalization()(model_3)
    model_3 = layers.LeakyReLU()(model_3)
    
    model_add_1 = layers.add([model_3,model_2])
    
    model_5 = layers.GRU(128,return_sequences=True,activation=None,go_backwards=True)(model_add_1)
    model_5 = layers.LeakyReLU()(model_5)
    model_5 = layers.GRU(32,return_sequences=True, activation=None,go_backwards=True)(model_5)
    model_5 = layers.LeakyReLU()(model_5)
    
    model_6 = layers.GRU(64,return_sequences=True,activation=None,go_backwards=True)(model_add_1)
    model_6 = layers.LeakyReLU()(model_6)
    model_6 = layers.GRU(32,return_sequences=True, activation=None,go_backwards=True)(model_6)
    model_6 = layers.LeakyReLU()(model_6)
    
    model_add_2 = layers.add([model_5,model_6,model_2_1])
    
    
    model_7 = layers.Dense(64, activation=None)(model_add_2)
    model_7 = layers.LeakyReLU()(model_7)
    model_7 = layers.Dropout(0.2)(model_7)
    model_7 = layers.Dense(16, activation=None)(model_7)
    model_7 = layers.LeakyReLU()(model_7)
    
    model_9 = layers.Dense(32, activation=None)(model_add_2)
    model_9 = layers.LeakyReLU()(model_9)
    model_9 = layers.Dropout(0.2)(model_9)
    model_9 = layers.Dense(16, activation=None)(model_9)
    model_9 = layers.LeakyReLU()(model_9)
    
    model_add_3 = layers.add([model_7,model_9])

    model_10 = layers.Dense(16, activation=None)(model_add_3)
    #model_10 = BatchNormalization()(model_10)
    model_10 = layers.LeakyReLU()(model_10)
    model_10 = layers.Dropout(0.5)(model_10)
    # model_10 = layers.Flatten()(model_10)
    #Model_7 = MaxPooling1D(pool_size=2)(mode)
    model_10 = layers.Flatten()(model_10)
    model_10 = layers.Dense(1, activation="softmax")(model_10)
    # delete above

    model = Model(inputs=i, outputs=model_10, name="conv2d")
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
    shape = (shape[1], shape[0])

    es_patience = params["ES_PATIENCE"]
    min_delta = params["MIN_DELTA"]

    six = bool(params["SIX"])
    concat = bool(params["CONCAT"])

    epoch_start = int(params["EPOCH_START"])
    target = int(params["TARGET"])

    initial_channels = params["INITIAL_CHANNELS"]
    # shape = shape + (initial_channels, )
    
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

    with tf.device('/CPU:0'): 

        train_file_name = train_file.split('/')[-1].split('.')[0]

        dataset_path = '../../data/txt_datasets/{}'.format(train_file_name)

        filenames, labels = process_bangladesh(six, dataset_path)

        samples = list(zip(filenames, labels))

        random.shuffle(samples)

        grouped_val_samples, grouped_train_samples = train_test_split(
                samples, test_size=train_test_ratio)

        random.shuffle(grouped_val_samples)
        random.shuffle(grouped_train_samples)

        if concat:
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
            val_dataset, train_dataset = process_data(val_samples, train_samples, generate_spec_custom, shape, batch_size, initial_channels)
        
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
            excel_dest = job_dir + "/validation_sheet_{}.xls".format(epoch)
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
        "--augm-params",
        help="Augmentation parameters",
        required=True,
    )
    args = parser.parse_args()
    arguments = args.__dict__
    job_dir = arguments.pop("job_dir")
    params = arguments.pop("params")
    params = json.loads(params)
    augm_params = arguments.pop("augm_params")
    augm_params = json.loads(augm_params)
    train_model(**arguments)
