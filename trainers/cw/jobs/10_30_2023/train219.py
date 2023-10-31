import sys

# sys.path.insert(0, 'main/models/conv')
from modules.main import parameters
from modules.main.helpers import *
from modules.main.global_helpers import *

from modules.audio_loader.JordanAudioLoader import JordanAudioLoader
from modules.audio_loader.IcbhiAudioLoader import IcbhiAudioLoader
from modules.audio_loader.BdAudioLoader import BdAudioLoader
from modules.audio_loader.PerchAudioLoader import PerchAudioLoader
from modules.audio_loader.helpers import (
    default_get_filenames,
    bd_get_filenames,
    perch_get_filenames,
)

from modules.spec_generator.SpecGenerator import SpecGenerator

from modules.callbacks.NewCallback2 import NewCallback2
from modules.callbacks.visualizationCallback import visualizationCallback

from modules.models.leaf_model9 import leaf_model9

from modules.parse_functions.parse_functions import spec_parser, generate_timed_spec

from modules.augmenter.functions import stretch_time, shift_pitch
from modules.models.core import Distiller

from collections import Counter
from sklearn.utils import class_weight
import tensorflow as tf

# from tf.keras.utils import to_categorical
import tensorflow_addons as tfa
import leaf_audio.frontend as leaf_frontend
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Optional

from tensorflow.python.client import device_lib


def train_model(
    datasets,
    model_to_be_trained,
    spec_aug_params,
    audio_aug_params,
    parse_function,
    label_passed,
):

    # simply initialize audio loader object for each dataset
    # mandatory parameters:  (1) root of dataset (2) function for extracting filenames
    # optional parameters: or other custom parameters, like the Bangladesh excel path
    # NOTE: name attribute: to distinguish between datasets when the same audio loader object is used for different datasets, such as antwerp and icbhi that both use IcbhiAudioLoader

    audio_loaders = []

    if datasets["Jordan"]:
        audio_loaders.append(
            JordanAudioLoader(parameters.jordan_root, default_get_filenames)
        )
    if datasets["Bd"]:
        audio_loaders.append(
            BdAudioLoader(parameters.bd_root, bd_get_filenames, parameters.excel_path)
        )
    if datasets["Perch"]:
        audio_loaders.append(
            PerchAudioLoader(parameters.perch_root, perch_get_filenames)
        )
    if datasets["Icbhi"]:
        audio_loaders.append(
            IcbhiAudioLoader(parameters.icbhi_root, default_get_filenames)
        )
    if datasets["Ant"]:
        # TODO: pass names?
        ant_loader = IcbhiAudioLoader(parameters.ant_root, default_get_filenames)
        ant_loader.name = "Antwerp"
        audio_loaders.append(ant_loader)
    # if datasets["SimAnt"]:
    #     sim_ant_loader =
    #     audio_loaders.append(PerchAudioLoader(bd_root, bd_get_filenames, excel_path, mode=parameters.mode))

    # this functions loads the audios files from the given input, often .wav and .txt files
    # input: [filename1, filename2, ...]
    # output: {'Icbhi': [[audio1, label2, filename1], [audio2, label2, filename2], 'Jordan:' : ... }
    audios_dict = load_audios(audio_loaders)
    # print(audios_dict)

    # ths function takes the full audios and prepares its N chunks accordingly
    # by default, it returns samples grouped by patient according to the respective logics of datasets
    # input: [[audio1, label1, filename1], [audio2, label2, filename2], ...]
    # output: [ [all chunks = [audio, label, filename] of all files for patient1], [same for patient 2], ...]
    audios_c_dict = prepare_audios(audios_dict)

    icbhi_dict = audios_c_dict.pop("Icbhi")

    # NOTE: # Data is grouped by dataset and patient thus far
    # this functions (1) splits each dataset into train and validation, then (2) after split, we don't care about grouping by patient = flatten to list of audios by patients to give a list of audios
    #  input: Full Dictionary:  {Icbhi: [] -> data grouped by PATIENT, Jordan: [] -> data grouped by PATIENT, ...}
    # output: Training /// Val  dictionary:   {Icbhi: [] -> data organized INDIVIDUALLY, Jordan: [] -> data organized  INDIVIDUALLY}
    train_audios_c_dict, val_audios_c_dict = split_and_extend(
        audios_c_dict, parameters.train_test_ratio, kfold=parameters.kfold
    )

    for i in range(len(train_audios_c_dict)):
        initialize_job()

        # NOTE: # Data is only grouped by dataset now
        # simplest step: now that everything is ready, we convert to spectrograms! it's the most straightforward step...
        _val_audios_c_dict = val_audios_c_dict[i]
        _train_audios_c_dict = train_audios_c_dict[i]

        # convert: [audio, label, filename] -> [SPEC, label, filename]
        val_samples = generate_audio_samples(_val_audios_c_dict)

        # ... but it's different for training because of augmentation. the following function sets up and merges 2 branches:
        #   1) augment AUDIO and convert to spectrogram
        #   2) convert to spectrogram and augment SPECTROGRAM
        # train_samples, original_training_length = set_up_training_samples(train_audios_c_dict, spec_aug_params, audio_aug_params)
        train_samples = generate_audio_samples(_train_audios_c_dict)

        _val_samples = [item for sublist in val_samples for item in sublist]
        _train_samples = [item for sublist in train_samples for item in sublist]

        if parameters.oversample:
            _train_samples = oversample(_train_samples)

        # adding icbhi official split
        icbhi_train_samples, icbhi_val_samples = return_official_icbhi_split(icbhi_dict)

        print("before icbhi addition")
        print(len(_train_samples))
        _train_samples = _train_samples + icbhi_train_samples
        print("after icbhi addition")
        print(len(_train_samples))

        np.random.shuffle(_train_samples)
        np.random.shuffle(_val_samples)
        # NOTE: # Data is NOT LONGER grouped by dataset

        # from now on it's cake!

        train_dataset, __, train_labels, __ = create_tf_dataset(
            _train_samples,
            batch_size=parameters.batch_size,
            shuffle=True,
            parse_func=parse_function,
        )
        val_dataset, val_specs, val_labels, val_filenames = create_tf_dataset(
            _val_samples, batch_size=1, shuffle=False, parse_func=parse_function
        )  # keep shuffle = False!
        train_non_pneumonia_nb, train_pneumonia_nb = (
            train_labels.count(0),
            train_labels.count(1),
        )

        print("-----------------------")
        print_dataset(train_labels, val_labels)

        # callbacks
        metrics_callback = NewCallback2(
            val_dataset, val_filenames, target_key="icbhi_score"
        )
        icbhi_val_dataset, _, _, icbhi_val_filenames = create_tf_dataset(
            icbhi_val_samples, batch_size=1, shuffle=False, parse_func=parse_function
        )  # keep shuffle = False!
        icbhi_metrics_callback = NewCallback2(
            icbhi_val_dataset, icbhi_val_filenames, target_key="icbhi_score"
        )
        val_samples_copy = _val_samples.copy()
        np.random.shuffle(val_samples_copy)
        samples = val_samples_copy[:25]
        viz_callback = visualizationCallback(samples)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=parameters.job_dir
        )

        # weights
        weights = None

        print("Initializing weights...")
        weights = []
        weights = class_weight.compute_class_weight(
            "balanced",
            [0, 1, 2, 3],
            [metrics_callback.convert(l) for l in train_labels],
        )
        weights = {i: weights[i] for i in range(0, len(weights))}
        print("weights = {}".format(weights))
        parameters.weights = list(weights.values())

        # teacher
        teacher = model_to_be_trained(_frontend=parameters.frontend)
        shape = (None, parameters.audio_length * parameters.sr)
        teacher.build(shape)

        opt = tf.keras.optimizers.SGD(
            learning_rate=parameters.lr, momentum=0.9, nesterov=False, name="SGD"
        )

        loss = tf.keras.losses.BinaryCrossentropy(
            label_smoothing=parameters.label_smoothing
        )

        teacher.build(shape)

        teacher.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

        teacher.summary(line_length=110)

        history = teacher.fit(
            train_dataset,
            epochs=parameters.n_epochs,
            verbose=2,
            # class_weight=weights,
            callbacks=[
                metrics_callback,
                icbhi_metrics_callback,
                viz_callback,
                tensorboard_callback,
            ],
        )

        plot_metrics(history)

        print("End")
        print("####################################")


def launch_job(
    datasets, model, spec_aug_params, audio_aug_params, parse_function, label_passed
):
    """
    parameters: 
    # dictionary:  datasets to use, 
    # model:  imported from modules/models.py, 
    # augmentation parameters:  (No augmentation if empty list is passed), 
    # parse function:  (inside modules/parsers) used to create tf.dataset object in create_tf_dataset
    """
    # in a given file named train$ (parent/cache folder named train$), we can have multiple jobs (child folders named 1,2,3)
    initialize_job()  #  initialize each (child) job inside the file (i.e, creates all the subfolders like tp/tn/gradcam/etc, file saving conventions, etc)
    print("Job dir: {}".format(parameters.job_dir))
    train_model(
        datasets, model, spec_aug_params, audio_aug_params, parse_function, label_passed
    )


if __name__ == "__main__":

    print("Tensorflow Version: {}".format(tf.__version__))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
    seed_everything()  # seeding np, tf, etc
    arguments = parameters.parse_arguments()
    parameters.init()
    parameters.mode = "pneumonia" if arguments["mode"] == "main" else arguments["mode"]
    parameters.n_classes = 2 if parameters.mode == "cw" else 1
    print(parameters.cache_root)
    print(parameters.mode)
    print(os.path.basename(__file__).split(".")[0])
    parameters.file_dir = os.path.join(
        parameters.cache_root, parameters.mode, os.path.basename(__file__).split(".")[0]
    )
    parameters.description = arguments["description"]
    print("Description: {}".format(parameters.description))

    parameters.n_epochs = 100

    testing_mode(
        int(arguments["testing"])
    )  # if true, adds _testing to the cache folder, sets a small number of epochs, smaller dataset, turn off a few settings in the callback, etc
    # works to set up the parent folder (i.e., overwrites if already exists) + works well with initialize_job above
    initialize_file_folder()
    print("-----------------------")

    ###### set up used for spec input models (default)
    parameters.hop_length = 254
    parameters.shape = (80, 500, 3)
    parameters.n_sequences = 9

    parameters.early_stopping = False
    parameters.es_patience = 25

    parameters.adaptive_lr = False
    parameters.lr_patience = 5
    parameters.min_lr = 1e-5
    parameters.factor = 0.5

    parameters.batch_size = 64
    parameters.weight_decay = 1e-4

    parameters.overlap_threshold = 0.15
    parameters.audio_length = 5
    parameters.step_size = 2.5

    parameters.class_weights = False
    parameters.activate_spectral_loss = False
    parameters.normalize = False
    parameters.stacking = False
    parameters.oversample = False
    parameters.one_hot_encoding = False
    parameters.activation = "softmax"
    parameters.n_filters = 80
    parameters.sr = 16000

    # parameters.mode = "cw"
    parameters.n_classes = 4
    spec_aug_params = []
    audio_aug_params = []

    ################################

    parameters.early_stopping = True
    parameters.es_patience = 13

    parameters.sr = 8000
    parameters.weight_decay = 1e-4
    parameters.batch_size = 64

    parameters.adaptive_lr = True
    parameters.lr = 5e-3
    parameters.lr_patience = 6
    parameters.factor = 0.25

    parameters.overlap_threshold = 0.3

    parameters.n_classes = 2
    parameters.activation = "sigmoid"
    parameters.code = -1

    parameters.kfold = True
    parameters.class_weights = True
    parameters.distillation = False
    parameters.viz_count = 5

    parameters.n_fft = 2048
    parameters.window_len = 100
    parameters.window_stride = 25
    parameters.shape = (80, 200, 3)
    parameters.icbhi_root = os.path.join(
        parameters.data_root, "raw_audios/icbhi_preprocessed_v2_8000/"
    )

    print("training parameters")
    print(list(parameters.__dict__.items())[15:])

    parameters.frontend = leaf_frontend.MelFilterbanks(
        sample_rate=parameters.sr,
        n_filters=parameters.n_filters,
        n_fft=parameters.n_fft,
        window_len=parameters.window_len,
        window_stride=parameters.window_stride,
        max_freq=float(parameters.sr / 2),
    )

    parameters.model = "resnet"
    launch_job(
        {"Bd": 0, "Jordan": 0, "Icbhi": 1, "Perch": 1, "Ant": 1, "SimAnt": 0},
        leaf_model9,
        spec_aug_params,
        audio_aug_params,
        None,
        [1, 0],
    )
    print("Job dir: {}".format(parameters.job_dir))

    # parameters.model = "resnet"
    # launch_job({"Bd": 0, "Jordan": 0, "Icbhi": 1, "Perch": 0, "Ant": 0, "SimAnt": 0,}, leaf_model9, spec_aug_params, audio_aug_params, None, [1,0])
    # print("Job dir: {}".format(parameters.job_dir))

