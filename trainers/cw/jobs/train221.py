import sys

from soundofnavi.main import parameters as p
from soundofnavi.main.parameters import initialize_job
from soundofnavi.training.training import train_function
from soundofnavi.main.helpers import (
    default_get_filenames,
    print_dataset,
    convert_cw_labels,
)
from soundofnavi.dataset.icbhi_dataset import IcbhiDataset
from soundofnavi.models.leaf_pretrained import leaf_pretrained

import leaf_audio.frontend as leaf_frontend
from sklearn.utils import class_weight
import tensorflow as tf
import numpy as np
import os


def launch_job(
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
        #     icbhi_dict, p.official_labels_path
        # )
        # train_samples = train_samples + icbhi_train_samples

        np.random.shuffle(train_samples)
        np.random.shuffle(val_samples)

        val_samples = val_samples[:100]
        train_samples = train_samples[:100]

        if p.oversample:
            train_samples = oversample(train_samples)

        # from now on it's cake!
        train_dataset, train_elements, train_labels, train_filenames = create_tf_dataset(
            train_samples,
            batch_size=p.batch_size,
            shuffle=True,
            parse_func=p.parse_function,
        )
        val_dataset, val_elements, val_labels, val_filenames = create_tf_dataset(
            val_samples, batch_size=1, shuffle=False, parse_func=p.parse_function
        )  # keep shuffle = False!

        print_dataset(train_labels, val_labels)

        # callbacks
        # metrics_callback = NewCallback2(
        #     val_dataset, val_filenames, target_key="icbhi_score"
        # )

        # weights
        if p.use_class_weights:
            print("Initializing weights...")
            weights = class_weight.compute_class_weight(
                class_weight="balanced",
                classes=[0, 1, 2, 3],
                y=[convert_cw_labels(l) for l in train_labels],
            )
            weights = {i: weights[i] for i in range(0, len(weights))}
            print("weights = {}".format(weights))
            p.weights = list(weights.values())

        # # model
        model = model_to_be_trained(_frontend=p.frontend)
        model.build(p.audio_shape)
        model.summary()

        # optimizer = tf.keras.optimizers.SGD(
        #     learning_rate=p.lr, momentum=0.9, nesterov=False, name="SGD"
        # )
        optimizer = tf.keras.optimizers.legacy.SGD(
            learning_rate=p.lr, momentum=0.9, nesterov=False, name="SGD"
        )

        loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=p.label_smoothing)

        metrics = [ConfusionMatrixMetric()]

        tb_callback = tf.keras.callbacks.TensorBoard(os.path.join(p.job_dir, "logs"))
        tb_callback.set_model(model)

        train_writer = tf.summary.create_file_writer(
            os.path.join(p.job_dir, "logs/train")
        )
        val_writer = tf.summary.create_file_writer(
            os.path.join(p.job_dir, "logs/validation")
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

        if not p.kfold:
            break

        initialize_job()


if __name__ == "__main__":
    p.seed_everything()
    p.init(p.parse_arguments(), os.path.basename(__file__).split(".")[0])
    p.n_epochs = 10

    p.early_stopping = True
    p.es_patience = 5

    p.sr = 8000
    p.weight_decay = 1e-4
    p.batch_size = 2

    p.adaptive_lr = True
    p.lr = 5e-3
    p.lr_patience = 2
    p.factor = 0.25

    p.overlap_threshold = 0.3

    p.n_classes = 2
    p.activation = "sigmoid"
    # p.code = -1

    p.kfold = False
    p.use_class_weights = True
    p.distillation = False
    p.one_hot_encoding = False
    p.viz_count = 5

    p.audio_length = 5
    p.n_fft = 2048
    p.window_len = 100
    p.window_stride = 25
    p.n_filters = 80
    p.initial_channels = 3
    p.audio_shape = (None, p.audio_length * p.sr, 1)
    p.spec_shape = (p.n_filters, 500, p.initial_channels)
    p.icbhi_root = os.path.join(p.data_root, "raw_audios/icbhi_preprocessed_v2_8000/")

    p.frontend = leaf_frontend.MelFilterbanks(
        sample_rate=p.sr, n_filters=p.n_filters, max_freq=float(p.sr / 2)
    )
    # p.frontend = leaf_frontend.Leaf(
    #     sample_rate=p.sr,
    #     n_filters=p.n_filters,
    #     # n_fft=p.n_fft,
    #     window_len=p.window_len,
    #     window_stride=p.window_stride,
    # )

    p.model = "resnet"

    launch_job(
        {"Bd": 0, "Jordan": 0, "Icbhi": 1, "Perch": 1, "Ant": 1, "SimAnt": 0},
        leaf_pretrained,
        p.spec_aug_params,
        p.audio_aug_params,
    )
