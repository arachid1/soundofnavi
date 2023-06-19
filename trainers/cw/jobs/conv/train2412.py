# from .modules.helpers import *
# from .modules.generators import *
# from .modules.metrics import *
# from .modules.callbacks import *
# from .modules.pneumonia import *
# from .modules.parse_functions import *
# from .modules.augmentation import *
# from .core import *
from .modules.main import parameters
from .modules.main.helpers import *
from collections import Counter

from .modules.audio_loader.JordanAudioLoader import JordanAudioLoader
from .modules.audio_loader.IcbhiAudioLoader import IcbhiAudioLoader
from .modules.audio_loader.BdAudioLoader import BdAudioLoader
from .modules.audio_loader.PerchAudioLoader import PerchAudioLoader
from .modules.audio_loader.helpers import default_get_filenames, bd_get_filenames, perch_get_filenames

# from .modules.audio_preparer.helpers import *

from .modules.spec_generator.SpecGenerator import SpecGenerator

# from .modules.augmenter.helpers import *

from .modules.callbacks.NewCallback import NewCallback

from .models import mixednet, model9, time_series_model, audio_model

from .modules.parse_functions.parse_functions import generate_spec, generate_timed_spec

from .modules.augmenter.functions import stretch_time, shift_pitch 

# from .modules.main.global_helpers import visualize_spec, pad_sample

from sklearn.utils import class_weight
import tensorflow as tf

# from .models import *

def train_model(datasets, model_to_be_trained, spec_aug_params, audio_aug_params, parse_function):
    # logs_path = job_dir + "/logs/"
    jordan_root = '../../data/jwyy9np4gv-3/'
    icbhi_root = '../../data/raw_audios/icbhi_preprocessed_v2_cleaned_8000/'
    bd_root = '../../data/PCV_SEGMENTED_Processed_Files/'
    excel_path = "../../data/Bangladesh_PCV_onlyStudyPatients.xlsx"
    perch_root="../../data/raw_audios/perch_8000_10seconds/"

    # full audios
    audio_loaders = []
    
    if datasets["Jordan"]: audio_loaders.append(JordanAudioLoader(jordan_root, default_get_filenames, mode=parameters.mode))
    if datasets["Bd"]: audio_loaders.append(BdAudioLoader(bd_root, bd_get_filenames, excel_path, mode=parameters.mode))
    if datasets["Perch"]: audio_loaders.append(PerchAudioLoader(perch_root, perch_get_filenames, mode=parameters.mode))
    if datasets["Icbhi"]: audio_loaders.append(IcbhiAudioLoader(icbhi_root, default_get_filenames, mode=parameters.mode))
    # if datasets["Ant"]: audio_loaders.append(PerchAudioLoader(bd_root, bd_get_filenames, excel_path, mode=parameters.mode))
    # if datasets["SimAnt"]: audio_loaders.append(PerchAudioLoader(bd_root, bd_get_filenames, excel_path, mode=parameters.mode))

    audios_dict = load_audios(audio_loaders)

    # audio chunks
    audios_c_dict = prepare_audios(audios_dict)

    # split and extend
    train_audios_c_dict, val_audios_c_dict = split_and_extend(audios_c_dict, parameters.train_test_ratio)

    # val
    val_samples = generate_audio_samples(val_audios_c_dict)

    # train
    train_samples = generate_audio_samples(train_audios_c_dict)

    # tf datasets
    train_dataset, __, train_labels, __ = create_tf_dataset(train_samples, batch_size=parameters.batch_size, shuffle=True, parse_func=parse_function)
    val_dataset, val_specs, val_labels, val_filenames = create_tf_dataset(val_samples, batch_size=1, shuffle=False, parse_func=parse_function)
    train_non_pneumonia_nb, train_pneumonia_nb = train_labels.count(0), train_labels.count(1)
    print("-----------------------")
    print_dataset(train_labels, val_labels)

    # weights
    weights = None

    if bool(parameters.class_weights):
        print("Initializing weights...")
        weights = class_weight.compute_class_weight(
            "balanced", [0, 1], [l for l in train_labels if l == 0 or l == 1])
        weights = {i: weights[i] for i in range(0, len(weights))}
        print("weights = {}".format(weights))
    
    # callbacks
    metrics_callback = NewCallback(val_dataset, val_filenames)

    gpus = tf.config.experimental.list_logical_devices('GPU')

    # model setting
    model = model_to_be_trained(**parameters.return_model_params())

    model.summary(line_length=110)

    if len(gpus) > 1:
        print("You are using 2 GPUs while the code is set up for one only.")
        exit()

    # training
    model.fit(
        train_dataset,
        epochs=parameters.n_epochs,
        verbose=2,
        class_weight=weights,
        callbacks=[metrics_callback]
    )

    model.save(parameters.job_dir + "/model_{}.h5".format(parameters.n_epochs))

def launch_job(datasets, model, spec_aug_params, audio_aug_params, parse_function):
    initialize_job()
    train_model(datasets, model, spec_aug_params, audio_aug_params, parse_function)

if __name__ == "__main__":
    # print("Tensorflow Version: {}".format(tf.__version__))
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    seed_everything()
    _, params, _, _ = parameters.parse_arguments()
    parameters.init(params)
    if parameters.testing:
        parameters.n_epochs = 2
        parameters.train_test_ratio = 0.5
    initialize_file_folder()
    print("-----------------------")
    spec_aug_params = []
    audio_aug_params = []
    parameters.mode = "CW"
    parameters.class_weights = False
    parameters.shape = (80000, )
    parameters.n_classes = 2
    parameters.label_smoothing = 0.4
    launch_job({"Icbhi": 1, "Jordan": 1, "Perch": 1, "Ant": 0, "SimAnt": 0, "Bd": 0,}, audio_model, spec_aug_params, audio_aug_params, None)
    parameters.label_smoothing = 0
    parameters.lr = 1e-4
    launch_job({"Icbhi": 1, "Jordan": 1, "Perch": 1, "Ant": 0, "SimAnt": 0, "Bd": 0,}, audio_model, spec_aug_params, audio_aug_params, None)
    parameters.weight_decay = 1e-4
    launch_job({"Icbhi": 1, "Jordan": 1, "Perch": 1, "Ant": 0, "SimAnt": 0, "Bd": 0,}, audio_model, spec_aug_params, audio_aug_params, None)
    # parameters.hop_length = 254
    # parameters.shape = (128, 315)
    # parameters.n_sequences = 9
    # spec_aug_params = []
    # audio_aug_params = []
    # launch_job(time_series_model, spec_aug_params, audio_aug_params, generate_timed_spec)

