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

def train_model(train_file, job_dir, params, wav_params, spec_params, model_to_be_trained, job_count):
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
    adaptive_lr = int(params["ADAPTIVE_LR"])
    cuberooting = int(params["CUBEROOTING"])
    normalizing = int(params["NORMALIZING"])

    initial_channels = params["INITIAL_CHANNELS"]
    
    model_params = {
        "N_CLASSES": n_classes,
        "SR": sr,
        "BATCH_SIZE": batch_size,
        "LR": lr,
        "SHAPE": shape,
        "INITIAL_CHANNELS": initial_channels,
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

    jordan_dataset = int(params["JORDAN_DATASET"])
    pneumonia_only = int(params["PNEUMONIA_ONLY"])

    wav_add = bool(wav_params['ADD'])
    wav_quantity = int(wav_params['QUANTITY'])
    wav_path = str(wav_params["WAV_PATH"]).split(',')

    print("All variables have been collected.")

    train_test_ratio = 0.8

    filenames = []
    labels = []

    wandb_name = __file__.split('/')[-1].split('.')[0] + str('_id{}'.format(job_count))
    # print(wandb_name)
    run = wandb.init(project="tensorboard-integration", name=wandb_name, sync_tensorboard=False)

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

    filenames, labels = process_icbhi(six, dataset_path)

    # path = '../../data/txt_datasets/{}'.format(train_file_name)

    nb_pneumonia = labels.count(1)
    nb_non_pneumonia = labels.count(0)
    print("Pneumonia patients: {}, Non-pneumonia patients: {}".format(nb_pneumonia, nb_non_pneumonia))

    # new dataset
    new_dataset_path = '../../data/txt_datasets/all_sw_coch_preprocessed_v2_param_v44_augm_v0_cleaned_8000/'
    _list = os.path.join(new_dataset_path, 'aa_paths_and_labels.txt')

    if jordan_dataset:
        new_filenames = []
        new_labels = []
        with open(_list) as infile:
            for line in infile:
                elements = line.rstrip("\n").rsplit(',', 1)
                filename = elements[0]
                label = float(elements[1])
                if pneumonia_only:
                    if label == 0:
                        continue
                new_filenames.append(filename)
                new_labels.append(label)
        
        # print(new_filenames)
        # print(len(new_filenames))
        # print(sorted(new_filenames))
        jordan_samples = list(zip(new_filenames, new_labels))
        jordan_samples = sorted(jordan_samples)
        jordan_train_samples = jordan_samples[:(int(0.8*len(jordan_samples))+2)]
        jordan_val_samples = jordan_samples[(int(0.8*len(jordan_samples))+2):]
        # jordan_train_samples = jordan_samples[:int(0.8*len(jordan_samples))]
        # jordan_val_samples = jordan_samples[int(0.8*len(jordan_samples)):]
        # print(len(jordan_train_samples))
        # print(len(jordan_val_samples))
        # jordan_val_samples, jordan_train_samples = train_test_split(jordan_samples, test_size=train_test_ratio)
    
    nb_pneumonia = new_labels.count(1)
    nb_non_pneumonia = new_labels.count(0)
    print("Pneumonia patients: {}, Non-pneumonia patients: {}".format(nb_pneumonia, nb_non_pneumonia))

    samples = list(zip(filenames, labels))

    samples = sorted(samples)

    train_samples = samples[:int(0.8*len(samples))]
    val_samples = samples[int(0.8*len(samples)):]

    if jordan_dataset:
        train_samples += jordan_train_samples
        val_samples += jordan_val_samples

    random.shuffle(val_samples)
    random.shuffle(train_samples)

    if testing:
        train_samples = train_samples[:50]
        val_samples = val_samples[:50]
        n_epochs = 1
        spec_quantity = 10
        wav_quantity = 10

    indices_to_vis = random.sample(range(len(val_samples)), 45)
    samples_to_vis = [val_samples[i] for i in indices_to_vis]

    if concat:
        f = concatenate_specs
        val_dataset, train_dataset = process_data(grouped_val_samples, grouped_train_samples, f, shape, batch_size, initial_channels, cuberooting)
    else:
        print("-----------------------")
        train_samples = [list(sample) for sample in train_samples]
        original_length = len(train_samples)
        train_labels = [label for _, label in train_samples] 
        nb_train_pneumonia, nb_train_non_pneumonia = train_labels.count(1), train_labels.count(0)
        print("Number of train recordings: {} with pneumonia: {} and non-pneumonia: {}".format(original_length, nb_train_pneumonia, nb_train_non_pneumonia))

        if augmentation:
            train_samples, train_labels = manage_label_augmentations(train_samples, wav_add, spec_add, wav_quantity, spec_quantity, job_version, wav_path, time_masking, frequency_masking, loudness, dataset_path, label=1)
            train_samples, train_labels = manage_label_augmentations(train_samples, wav_add, spec_add, 600, 600, job_version, wav_path, time_masking, frequency_masking, loudness, dataset_path, label=0)

        val_labels = [label for _, label in val_samples] 
        nb_val_pneumonia, nb_val_non_pneumonia = val_labels.count(1), val_labels.count(0)
        print("Number of val recordings: {} with pneumonia: {} and non-pneumonia: {}".format(len(val_samples), nb_val_pneumonia, nb_val_non_pneumonia))
        print("-----------------------")

        # lr = 1e-3 * (original_length / len(train_samples))

        # print("The initial learning rate for this job is: {}".format(lr))
        f = generate_spec
        val_dataset, train_dataset = process_data(val_samples, train_samples, f, shape, batch_size, initial_channels, cuberooting, normalizing=normalizing)

        train_dataset = train_dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))

        mixup_quantity = 158
        alpha = 0.4
        both_classes = True

        if both_classes:
            pneumonia_train_dataset = train_dataset.filter(lambda x, y: y == 1)
            non_pneumonia_train_dataset = train_dataset.filter(lambda x, y: y == 0)

            train_slice_one = pneumonia_train_dataset.take(mixup_quantity) 
            train_slice_two = non_pneumonia_train_dataset.take(mixup_quantity)
        else:
            train_slice_one = train_dataset.take(mixup_quantity) 
            train_slice_two = train_dataset.take(mixup_quantity)
        
        train_slice_one = train_slice_one.shuffle(mixup_quantity)
        train_slice_two = train_slice_two.shuffle(mixup_quantity)

        train_slice = tf.data.Dataset.zip((train_slice_one, train_slice_two))

        mixup_batch_size = 1

        mixup_train_dataset = train_slice.map(lambda ds_one, ds_two: mix_up(ds_one, ds_two, batch_size=mixup_batch_size, alpha=alpha), num_parallel_calls=tf.data.AUTOTUNE)

        train_dataset = train_dataset.concatenate(mixup_train_dataset)

        mixup_length = len(list(train_dataset.as_numpy_iterator()))
        print("New dataset length after mix-up augmentation: {}".format(mixup_length))

        train_dataset = train_dataset.shuffle(mixup_length)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(1)

    # weights
    weights = None

    if bool(params["CLASS_WEIGHTS"]):
        print("Initializing weights...")
        weights = class_weight.compute_class_weight(
            "balanced", np.unique(labels), labels)
        weights = {i: weights[i] for i in range(0, len(weights))}
        print("weights = {}".format(weights))
    
    # callbacks
    metrics_callback = metric_callback(val_dataset, shape, initial_channels, n_classes, job_dir, sr, min_delta, es_patience, patience, min_lr, factor, epoch_start, target, job_count, clause, samples_to_vis, f, cuberooting, normalizing, adaptive_lr=adaptive_lr)

    gpus = tf.config.experimental.list_logical_devices('GPU')

    # model setting
    model = model_to_be_trained(**model_params)

    model.summary(line_length=110)

    if len(gpus) > 1:
        print("You are using 2 GPUs while the code is set up for one only.")
        exit()
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=n_epochs,
        verbose=2,
        class_weight=weights,
        callbacks=[metrics_callback]
    )

    model.save(job_dir + "/{}/model_{}.h5".format(job_count, n_epochs))

    run.finish()

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
    # config = ConfigProto()
    # config.gpu_options.allow_growth = True
    # session = InteractiveSession(config=config)
    print("-----------------------")
    print("Using module: {}".format(__file__[-15:]))
    print("Using train_file: {}".format(train_file))
    print("Job directory: {}".format(job_dir))
    print("Parameters: {}".format(params))
    print("Augmentation parameters for audio: {}".format(wav_params))
    print("Augmentation parameters for spectrograms: {}".format(spec_params))
    print("-----------------------")
    train_model(train_file, job_dir, params, wav_params, spec_params, mixednet, job_count=0)
    # print("Model 9 with new params")
    # params['WEIGHT_DECAY'] = 1e-5
    # print("Parameters: {}".format(params))
    # train_model(train_file, job_dir, params, wav_params, spec_params, model9, job_count=2)
    # print("-----------------------")
    # print("Model 9 with new params")
    # params['WEIGHT_DECAY'] = 1e-3 # back to original
    # params['LABEL_SMOOTHING'] = 0.4 
    # print("Parameters: {}".format(params))
    # train_model(train_file, job_dir, params, wav_params, spec_params, model9, job_count=3)
    # print("-----------------------")
