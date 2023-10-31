from tensorflow.python.client import device_lib
import sys
# sys.path.insert(0, 'main/models/conv')
from modules.main import parameters
from modules.main.parameters import initialize_job
from modules.main.training import *
from modules.main.helpers import *
from modules.main.global_helpers import *

from modules.audio_loader.JordanAudioLoader import JordanAudioLoader
from modules.audio_loader.IcbhiAudioLoader import IcbhiAudioLoader
from modules.audio_loader.BdAudioLoader import BdAudioLoader
from modules.audio_loader.PerchAudioLoader import PerchAudioLoader
from modules.audio_loader.helpers import default_get_filenames, bd_get_filenames, perch_get_filenames

from modules.spec_generator.SpecGenerator import SpecGenerator

from modules.callbacks.NewCallback2 import NewCallback2
from modules.callbacks.visualizationCallback import visualizationCallback

from modules.models.leaf_pretrained import leaf_pretrained

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
import os
# import warnings
# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


def snr_function(codes, samples):
    snrs = []
    snrs_dataset_dict = defaultdict(lambda: [])
    snrs_mic_dict = defaultdict(lambda: [])
    for i, data in enumerate(codes):
        audio = samples[i][0]
        snr = signaltonoise(audio)
        # print(snr)
        # snr = np.array(snr, dtype=np.float32)
        snrs.append(snr)
        # print(val_snrs)
        snrs_dataset_dict[data[1]].append(snr)
        snrs_mic_dict[data[2]].append(snr)
    return snrs, snrs_dataset_dict, snrs_mic_dict


def train_model(datasets, model_to_be_trained, spec_aug_params, audio_aug_params):

    audio_loaders = []

    if datasets["Jordan"]:
        audio_loaders.append(JordanAudioLoader(
            parameters.jordan_root, default_get_filenames))
    if datasets["Bd"]:
        audio_loaders.append(BdAudioLoader(
            parameters.bd_root, bd_get_filenames, parameters.excel_path))
    if datasets["Perch"]:
        audio_loaders.append(PerchAudioLoader(
            parameters.perch_root, perch_get_filenames))
    if datasets["Icbhi"]:
        audio_loaders.append(IcbhiAudioLoader(
            parameters.icbhi_root, default_get_filenames))
    if datasets["Ant"]:
        # TODO: pass names?
        ant_loader = IcbhiAudioLoader(
            parameters.ant_root, default_get_filenames)
        ant_loader.name = "Antwerp"
        audio_loaders.append(ant_loader)
    # if datasets["SimAnt"]:
    #     sim_ant_loader =
    #     audio_loaders.append(PerchAudioLoader(bd_root, bd_get_filenames, excel_path, mode=parameters.mode))

    # input: [filename1, filename2, ...]
    # output: dictionnaries of FULL audios by dataset
    # {'Icbhi': [[audio1, label2, filename1], [audio2, label2, filename2]], 'Jordan:' : ... }
    audios_dict = load_audios(audio_loaders)

    # input: ['Icbhi': [[audio1, label, filename1], [audio2, label, filename2]], 'Jordan:' : ...]
    # output: dictionaries of chunks of FULL audios, organized by patient and dataset
    # [ 'Icbhi': [chunks for patient 1], [chunks for patient 2], 'Jordan:' : ...]
    audios_c_dict = prepare_audios(audios_dict)
    icbhi_dict = audios_c_dict.pop('Icbhi')
    rando = audios_c_dict.pop('Perch')
    # rando = audios_c_dict.pop('Antwerp')

    # NOTE: # Data is grouped by dataset and patient thus far i.e. {'Icbhi': [[patient1], [patient2], ...}
    train_audios_c_dict, val_audios_c_dict = split_and_extend(
        audios_c_dict, parameters.train_test_ratio, kfold=True)
    # NOTE: Data is returned as separate train/val dictionary accessible with the same index
    # chunks no longer organized by patient

    for i in range(len(train_audios_c_dict)):

        _val_audios_c_dict = val_audios_c_dict[i]
        _train_audios_c_dict = train_audios_c_dict[i]

        val_samples = generate_audio_samples(_val_audios_c_dict)
        train_samples = generate_audio_samples(_train_audios_c_dict)

        # adding icbhi official split
        # icbhi_train_samples, icbhi_val_samples = return_official_icbhi_split(
        #     icbhi_dict, parameters.official_labels_path)
        # train_samples = train_samples + icbhi_train_samples
        icbhi_train_samples, icbhi_val_samples = return_official_icbhi_split(
            icbhi_dict, parameters.official_labels_path)
        train_samples = icbhi_train_samples
        val_samples = icbhi_val_samples

        np.random.shuffle(train_samples)
        np.random.shuffle(val_samples)

        if parameters.oversample:
            train_samples = oversample(train_samples)

        # val_codes = codify(val_samples)
        train_codes = codify(train_samples)
        exit()

        val_snrs, val_snrs_dataset_dict, val_snrs_mic_dict = snr_function(
            val_codes, val_samples)

        train_snrs, train_snrs_dataset_dict, train_snrs_mic_dict = snr_function(
            train_codes, train_samples)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)
        plt.boxplot(np.array([train_snrs, val_snrs], dtype=object),
                    labels=["train", "val"])
        plt.savefig(os.path.join(parameters.job_dir,
                                 "others/train_val"))
        plt.close()

        print(train_snrs_mic_dict.keys())
        print(len(train_snrs_mic_dict[2]))
        print(val_snrs_mic_dict.keys())
        exit()
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)
        plt.boxplot(np.array([train_snrs_mic_dict[1], train_snrs_mic_dict[3], train_snrs_mic_dict[4], val_snrs_mic_dict[1], val_snrs_mic_dict[3]], dtype=object),
                    labels=["Train_Meditron", "Train_LittC2SE", "Train_AKGC417L", "Val_Meditron", "Val_"])
        plt.savefig(os.path.join(parameters.job_dir, "others/x"))
        plt.close()

        exit()

        domain_examples = val_samples[:20]
        domain_examples_codes = val_codes[:20]

        # from now on it's cake!
        train_dataset, __, train_labels, __ = create_tf_dataset(
            train_samples, batch_size=parameters.batch_size, shuffle=True, parse_func=parameters.parse_function)
        val_dataset, val_specs, val_labels, val_filenames = create_tf_dataset(
            val_samples, batch_size=1, shuffle=False, parse_func=parameters.parse_function)  # keep shuffle = False!
        domain_dataset, _, _, _ = create_tf_dataset(
            domain_examples, batch_size=1, shuffle=False, parse_func=parameters.parse_function)  # keep shuffle = False!

        print_dataset(train_labels, val_labels)

        # # callbacks
        metrics_callback = NewCallback2(
            val_dataset, val_filenames, target_key="icbhi_score")

        # # weights
        if parameters.use_class_weights:
            print("Initializing weights...")
            weights = class_weight.compute_class_weight(
                "balanced", [0, 1, 2, 3], [metrics_callback.convert(l) for l in train_labels])
            weights = {i: weights[i] for i in range(0, len(weights))}
            print("weights = {}".format(weights))
            parameters.weights = list(weights.values())

        # # teacher
        teacher = model_to_be_trained(_frontend=parameters.frontend)
        teacher.build(parameters.audio_shape)
        # teacher.__call__(next(iter(train_dataset)), training=False)
        teacher.summary()

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=parameters.lr, momentum=0.9, nesterov=False, name="SGD"
        )

        loss_fn = tf.keras.losses.BinaryCrossentropy(
            label_smoothing=parameters.label_smoothing)

        metrics = [ConfusionMatrixMetric()]

        tb_callback = tf.keras.callbacks.TensorBoard(
            os.path.join(parameters.job_dir, "logs"))
        tb_callback.set_model(teacher)

        train_writer = tf.summary.create_file_writer(
            os.path.join(parameters.job_dir, "logs/train"))
        val_writer = tf.summary.create_file_writer(
            os.path.join(parameters.job_dir, "logs/validation"))

        teacher.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics
        )

        # teacher.fit(
        #     train_dataset,
        #     validation_data=val_dataset,
        #     epochs=parameters.n_epochs,
        #     verbose=2,
        #     callbacks=metrics_callback
        # )

        # teacher = model_to_be_trained(_frontend=parameters.frontend)
        # shape = (None, parameters.audio_length*parameters.sr)
        # teacher.build(shape)

        # for l in teacher.get_layer('sequential').get_layer('resnet50').layers:
        #     print(l.name)
        # exit()

        train_function(teacher, loss_fn, optimizer, train_dataset,
                       val_dataset, train_writer, val_writer, domain_dataset)

        if not parameters.kfold:
            break
        initialize_job()


if __name__ == "__main__":

    parameters.seed_everything()
    parameters.init(parameters.parse_arguments(),
                    os.path.basename(__file__).split('.')[0])
    parameters.n_epochs = 10

    parameters.early_stopping = True
    parameters.es_patience = 5

    parameters.sr = 8000
    parameters.weight_decay = 1e-4
    parameters.batch_size = 2

    parameters.adaptive_lr = True
    parameters.lr = 5e-3
    parameters.lr_patience = 2
    parameters.factor = 0.25

    parameters.overlap_threshold = 0.3

    parameters.n_classes = 2
    parameters.activation = "sigmoid"
    # parameters.code = -1

    parameters.kfold = False
    parameters.use_class_weights = True
    parameters.distillation = False
    parameters.one_hot_encoding = False
    parameters.viz_count = 5

    parameters.audio_length = 5
    parameters.n_fft = 2048
    parameters.window_len = 100
    parameters.window_stride = 25
    parameters.audio_shape = (None, parameters.audio_length*parameters.sr)
    parameters.spec_shape = (80, 200, 3)
    parameters.n_filters = 80
    parameters.icbhi_root = os.path.join(
        parameters.data_root, 'raw_audios/icbhi_preprocessed_v2_8000/')

    parameters.frontend = leaf_frontend.Leaf(
        sample_rate=parameters.sr,
        n_filters=parameters.n_filters,
        # n_fft=parameters.n_fft,
        window_len=parameters.window_len,
        window_stride=parameters.window_stride,
        # max_freq=float(parameters.sr/2)
    )

    parameters.model = "resnet"

    train_model({"Bd": 0, "Jordan": 0, "Icbhi": 1, "Perch": 1, "Ant": 1, "SimAnt": 0, },
                leaf_pretrained, parameters.spec_aug_params, parameters.audio_aug_params)
    # print("Job dir: {}".format(parameters.job_dir))
    # parameters.model = "resnet"
    # launch_job({"Bd": 0, "Jordan": 0, "Icbhi": 1, "Perch": 0, "Ant": 0, "SimAnt": 0,}, leaf_model9, spec_aug_params, audio_aug_params, None, [1,0])
    # print("Job dir: {}".format(parameters.job_dir))
