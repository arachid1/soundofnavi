import sys

sys.path.insert(
    0, "/Users/alirachidi/Documents/Sonavi_Labs/classification_algorithm/trainers"
)
from modules.main import parameters
from modules.main.parameters import initialize_job
from modules.main.training import *
from modules.main.helpers import *
from modules.main.global_helpers import *
from modules.audio_loader.helpers import default_get_filenames
from modules.callbacks.NewCallback2 import NewCallback2
from modules.dataset.IcbhiDataset import IcbhiDataset
from modules.models.leaf_pretrained import leaf_pretrained

import leaf_audio.frontend as leaf_frontend
from sklearn.utils import class_weight
import tensorflow as tf
import numpy as np
import os


def train_model(
    datasets_selection, model_to_be_trained, spec_aug_params, audio_aug_params
):
    datasets = []
    # if datasets["Jordan"]:
    #         datasets.append(JordanDataset(
    #             p.jordan_root))
    # if datasets["Bd"]:
    #     datasets.append(BdDataset(
    #         p.bd_root, p.excel_path))
    # if datasets["Perch"]:
    #     datasets.append(PerchDataset(
    #         p.perch_root))
    if datasets_selection["Icbhi"]:
        icbhi_dataset = IcbhiDataset(
            "Icbhi", p.icbhi_root, p.icbhi_metadata_root, default_get_filenames
        )
        datasets.append(icbhi_dataset)

    # if datasets["Ant"]:
    #     ant_loader = AntwerpDataset(
    #         p.ant_root)
    #     ant_loader.name = "Antwerp"
    #     datasets.append(ant_loader)

    audios_dict = {}
    for d in datasets:
        d.load_recordings()
        d.prepare_slices()
        # TODO: write a custom flattening of dictionary function
        slices_by_patient = d.return_slices_by_patient()
        audios_dict[d.id] = slices_by_patient

    # icbhi_dict = audios_dict.pop("Icbhi")  # popping to use with official split

    train_audios_dict, val_audios_dict = split(
        audios_dict, p.train_test_ratio, kfold=True
    )

    for i in range(len(train_audios_dict)):
        # looping through the kfold pairs of train/val datasets
        _val_audios_dict = val_audios_dict[i]
        _train_audios_dict = train_audios_dict[i]

        val_samples = generate_audio_samples(_val_audios_dict)
        train_samples = generate_audio_samples(_train_audios_dict)

        # icbhi_train_samples, icbhi_val_samples = return_official_icbhi_split(
        #     icbhi_dict, parameters.official_labels_path
        # )
        # train_samples = train_samples + icbhi_train_samples

        np.random.shuffle(train_samples)
        np.random.shuffle(val_samples)

        if parameters.oversample:
            train_samples = oversample(train_samples)

        # from now on it's cake!
        train_dataset, __, train_labels, __ = create_tf_dataset(
            train_samples,
            batch_size=parameters.batch_size,
            shuffle=True,
            parse_func=parameters.parse_function,
        )
        val_dataset, val_specs, val_labels, val_filenames = create_tf_dataset(
            val_samples,
            batch_size=1,
            shuffle=False,
            parse_func=parameters.parse_function,
        )  # keep shuffle = False!

        print_dataset(train_labels, val_labels)

        # callbacks
        # metrics_callback = NewCallback2(
        #     val_dataset, val_filenames, target_key="icbhi_score"
        # )

        # weights
        if parameters.use_class_weights:
            print("Initializing weights...")
            weights = class_weight.compute_class_weight(
                class_weight="balanced",
                classes=[0, 1, 2, 3],
                y=[NewCallback2.convert(l) for l in train_labels],
            )
            weights = {i: weights[i] for i in range(0, len(weights))}
            print("weights = {}".format(weights))
            parameters.weights = list(weights.values())

        # # model
        # print()
        model = model_to_be_trained(_frontend=parameters.frontend)
        model.build(parameters.audio_shape)
        model.summary()
        print("here")
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=parameters.lr, momentum=0.9, nesterov=False, name="SGD"
        )

        loss_fn = tf.keras.losses.BinaryCrossentropy(
            label_smoothing=parameters.label_smoothing
        )

        metrics = [ConfusionMatrixMetric()]

        tb_callback = tf.keras.callbacks.TensorBoard(
            os.path.join(parameters.job_dir, "logs")
        )
        tb_callback.set_model(model)

        train_writer = tf.summary.create_file_writer(
            os.path.join(parameters.job_dir, "logs/train")
        )
        val_writer = tf.summary.create_file_writer(
            os.path.join(parameters.job_dir, "logs/validation")
        )

        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

        train_function(
            model,
            loss_fn,
            optimizer,
            train_dataset,
            val_dataset,
            train_writer,
            val_writer,
        )

        if not parameters.kfold:
            break

        initialize_job()


if __name__ == "__main__":
    parameters.seed_everything()
    parameters.init(
        parameters.parse_arguments(), os.path.basename(__file__).split(".")[0]
    )
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
    parameters.audio_shape = (None, parameters.audio_length * parameters.sr, None)
    parameters.spec_shape = (80, 200, 3)
    parameters.n_filters = 80
    parameters.icbhi_root = os.path.join(
        parameters.data_root, "raw_audios/icbhi_preprocessed_v2_8000/"
    )

    parameters.frontend = leaf_frontend.MelFilterbanks(
        sample_rate=parameters.sr,
        n_filters=parameters.n_filters,
        max_freq=float(parameters.sr / 2),
    )
    # parameters.frontend = leaf_frontend.Leaf(
    #     sample_rate=parameters.sr,
    #     n_filters=parameters.n_filters,
    #     # n_fft=parameters.n_fft,
    #     window_len=parameters.window_len,
    #     window_stride=parameters.window_stride,
    # )

    parameters.model = "resnet"

    train_model(
        {"Bd": 0, "Jordan": 0, "Icbhi": 1, "Perch": 1, "Ant": 1, "SimAnt": 0},
        leaf_pretrained,
        parameters.spec_aug_params,
        parameters.audio_aug_params,
    )
