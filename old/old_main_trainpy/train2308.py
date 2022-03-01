from .modules.helpers import *
from .modules.generators import *
from .modules.metrics import *
from .modules.callbacks import *
from .modules.pneumonia import *
from .modules.parse_functions import *
from .modules.augmentation import *
from .core import *
from .models import *

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
from .models import *

# @tf.function
def get_box(lambda_value, shape):
    cut_rat = tf.math.sqrt(1.0 - lambda_value)

    cut_w = shape[1] * cut_rat  # rw
    cut_w = tf.cast(cut_w, tf.int32)

    cut_h = shape[0] * cut_rat  # rh
    cut_h = tf.cast(cut_h, tf.int32)

    cut_x = tf.random.uniform((1,), minval=0, maxval=shape[1], dtype=tf.int32)  # rx
    cut_y = tf.random.uniform((1,), minval=0, maxval=shape[0], dtype=tf.int32)  # ry

    boundaryx1 = tf.clip_by_value(cut_x[0] - cut_w // 2, 0, shape[1])
    boundaryy1 = tf.clip_by_value(cut_y[0] - cut_h // 2, 0, shape[0])

    bbx2 = tf.clip_by_value(cut_x[0] + cut_w // 2, 0, shape[1])
    bby2 = tf.clip_by_value(cut_y[0] + cut_h // 2, 0, shape[0])

    target_h = bby2 - boundaryy1
    if target_h == 0:
        target_h += 1

    target_w = bbx2 - boundaryx1
    if target_w == 0:
        target_w += 1

    # tf.print("here")
    # tf.print(boundaryx1)
    # tf.print(boundaryy1)
    # tf.print(target_h)
    # tf.print(target_w)
    
    return boundaryx1, boundaryy1, target_h, target_w

# @tf.function
def cutmix(train_ds_one, train_ds_two, shape):
    (image1, label1), (image2, label2) = train_ds_one, train_ds_two

    alpha = [0.25]
    beta = [0.25]

    # Get a sample from the Beta distribution
    # lambda_value = sample_beta_distribution(1, alpha, beta)
    lambda_value = tf.random.uniform(shape=[], minval=0.1, maxval=0.9)

    # Define Lambda
    # lambda_value = lambda_value[0]

    # Get the bounding box offsets, heights and widths
    boundaryx1, boundaryy1, target_h, target_w = get_box(lambda_value, shape)
    # tf.print("here")
    # tf.print(boundaryy1)
    if boundaryy1 >= 32:
        if boundaryy1 < 64:
            boundaryy1 = boundaryy1 - 32
        elif boundaryy1 < 96: 
            boundaryy1 = boundaryy1 - 64
        else:
            boundaryy1 = boundaryy1 - 96
    # tf.print(boundaryy1)
    # tf.print(target_h)
    # tf.print(target_w)

    # Get a patch from the second image (`image2`)
    crop2 = tf.image.crop_to_bounding_box(
        image2, boundaryy1, boundaryx1, target_h, target_w
    )
    # Pad the `image2` patch (`crop2`) with the same offset
    image2 = tf.image.pad_to_bounding_box(
        crop2, boundaryy1, boundaryx1, shape[0], shape[1]
    )
    # Get a patch from the first image (`image1`)
    crop1 = tf.image.crop_to_bounding_box(
        image1, boundaryy1, boundaryx1, target_h, target_w
    )
    # Pad the `image1` patch (`crop1`) with the same offset
    img1 = tf.image.pad_to_bounding_box(
        crop1, boundaryy1, boundaryx1, shape[0], shape[1]
    )

    # Modify the first image by subtracting the patch from `image1`
    # (before applying the `image2` patch)
    image1 = image1 - img1
    # Add the modified `image1` and `image2`  together to get the CutMix image
    image = image1 + image2

    # Adjust Lambda in accordance to the pixel ration
    lambda_value = 1 - (target_w * target_h) / (shape[0] * shape[1])
    lambda_value = tf.cast(lambda_value, tf.float32)

    # Combine the labels of both images
    label = lambda_value * label1 + (1 - lambda_value) * label2
    # return image1, image2, image, label
    return image, label

def apply_cutmix_augmentation(train_dataset, both_classes, repeat_factor, cutmix_quantity, shape):

    if both_classes:
        train_dataset = train_dataset.filter(lambda x, y: tf.numpy_function(remove_zero_padded, [x, shape], tf.bool))
        print("length after removing zero-pads: " + str(len(list(train_dataset.as_numpy_iterator()))))
        pneumonia_train_dataset = train_dataset.filter(lambda x, y: y == 1)
        non_pneumonia_train_dataset = train_dataset.filter(lambda x, y: y == 0)
        train_slice_one = pneumonia_train_dataset.take(158)
        train_slice_one = train_slice_one.repeat(repeat_factor) 
        train_slice_two = non_pneumonia_train_dataset.take(cutmix_quantity)
    else:
        train_slice_one = train_dataset.take(cutmix_quantity) 
        train_slice_two = train_dataset.take(cutmix_quantity)
    
    train_slice_one = train_slice_one.shuffle(cutmix_quantity)
    train_slice_two = train_slice_two.shuffle(cutmix_quantity)

    train_slice = tf.data.Dataset.zip((train_slice_one, train_slice_two))

    cutmix_train_dataset = train_slice.map(lambda ds_one, ds_two: cutmix(ds_one, ds_two, shape), num_parallel_calls=tf.data.AUTOTUNE)

    return cutmix_train_dataset

# def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
#     gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
#     gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
#     return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def train_model(train_file, job_dir, params, wav_params, spec_params, model_to_be_trained, job_count, pneumonia_augmix_quantity, non_pneumonia_augmix_quantity, repeat_factor):
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

        # if augmentation: 
        #     pneumonia_augmix_samples = apply_augmix_augmentation(train_samples, pneumonia_augmix_quantity, dataset_path, job_version, wav_path, 1, shape)
        #     non_pneumonia_augmix_samples = apply_augmix_augmentation(train_samples, non_pneumonia_augmix_quantity, dataset_path, job_version, wav_path, 0, shape)
        #     train_samples.extend(pneumonia_augmix_samples)
        #     train_samples.extend(non_pneumonia_augmix_samples)

        # print("Length of training samples after augmix augmentation: {}".format(len(train_samples)))

        val_labels = [label for _, label in val_samples] 
        nb_val_pneumonia, nb_val_non_pneumonia = val_labels.count(1), val_labels.count(0)
        print("Number of val recordings: {} with pneumonia: {} and non-pneumonia: {}".format(len(val_samples), nb_val_pneumonia, nb_val_non_pneumonia))
        print("-----------------------")

        f = generate_spec
        val_dataset, train_dataset = process_data(val_samples, train_samples, f, shape, batch_size, initial_channels, cuberooting, normalizing=normalizing)

        train_dataset = train_dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))

        cutmix_quantity = 158*repeat_factor
        alpha = 0.4
        both_classes = True

        if augmentation:
            cutmix_train_dataset = apply_cutmix_augmentation(train_dataset, both_classes, repeat_factor, cutmix_quantity, shape)

        # arr_full = list(cutmix_train_dataset.as_numpy_iterator())

        # for i in range(5):
        #     folder = 'cutmix_to_show/{}_32boxstart'.format(i)

        #     if not (os.path.exists(folder)):
        #         os.mkdir(folder)

        #     images_one = arr_full[i][0]
        #     images_one = np.squeeze(images_one)
        #     visualize_spectrogram(images_one, 8000, os.path.join(folder, 'images_one.png'))

        #     images_two = arr_full[i][1]
        #     images_two = np.squeeze(images_two)
        #     visualize_spectrogram(images_two, 8000, os.path.join(folder, 'images_two.png'))

        #     images_one_mod = arr_full[i][2]
        #     images_one_mod = np.squeeze(images_one_mod)
        #     visualize_spectrogram(images_one_mod, 8000, os.path.join(folder, 'cutmix.png'))

        # exit()

        train_dataset = train_dataset.concatenate(cutmix_train_dataset)
        
        cutmix_length = len(list(train_dataset.as_numpy_iterator()))

        print("New dataset length after cutmix augmentation: {}".format(cutmix_length))

        train_dataset = train_dataset.shuffle(cutmix_length)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(1)
        # exit()

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
    print("-----------------------")
    print("Using module: {}".format(__file__[-15:]))
    print("Using train_file: {}".format(train_file))
    print("Job directory: {}".format(job_dir))
    print("Parameters: {}".format(params))
    print("Augmentation parameters for audio: {}".format(wav_params))
    print("Augmentation parameters for spectrograms: {}".format(spec_params))
    print("-----------------------")
    train_model(train_file, job_dir, params, wav_params, spec_params, mixednet, 0, 158, 400, repeat_factor=1)
    # print("-----------------------")
    # print("repeat factor of 3")
    # train_model(train_file, job_dir, params, wav_params, spec_params, mixednet, 0, 158, 400, repeat_factor=3)
    # print("-----------------------")
    # print("158/800 with repeat factor of 4")
    # train_model(train_file, job_dir, params, wav_params, spec_params, mixednet, 0, 158, 800, repeat_factor=4)
    # print("-----------------------")
    # print("158/800 with v42, v43")
    # wav_params['WAV_PATH'] = 'v42,v43'
    # print(wav_params)
    # train_model(train_file, job_dir, params, wav_params, spec_params, mixednet, 2, 158, 1000, repeat_factor=4)
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
