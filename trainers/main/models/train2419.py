import sys
sys.path.insert(0, '/home/alirachidi/classification_algorithm/trainers/main/models/conv')
from modules.main import parameters
from modules.main.helpers import *

from modules.audio_loader.JordanAudioLoader import JordanAudioLoader
from modules.audio_loader.IcbhiAudioLoader import IcbhiAudioLoader
from modules.audio_loader.BdAudioLoader import BdAudioLoader
from modules.audio_loader.PerchAudioLoader import PerchAudioLoader
from modules.audio_loader.helpers import default_get_filenames, bd_get_filenames, perch_get_filenames

from modules.spec_generator.SpecGenerator import SpecGenerator

from modules.callbacks.NewCallback import NewCallback

from modules.models import mixednet, model9, time_series_model, kapre_model

from modules.parse_functions.parse_functions import spec_parser, generate_timed_spec

from modules.augmenter.functions import stretch_time, shift_pitch 

from collections import Counter
from sklearn.utils import class_weight
import tensorflow as tf

def train_model(datasets, model_to_be_trained, spec_aug_params, audio_aug_params, parse_function):

    # full audios
    audio_loaders = []
    
    # TODO: pass names and testing files as elements
    if datasets["Jordan"]: audio_loaders.append(JordanAudioLoader(parameters.jordan_root, default_get_filenames))
    if datasets["Bd"]: audio_loaders.append(BdAudioLoader(parameters.bd_root, bd_get_filenames, parameters.excel_path))
    if datasets["Perch"]: audio_loaders.append(PerchAudioLoader(parameters.perch_root, perch_get_filenames))
    if datasets["Icbhi"]: audio_loaders.append(IcbhiAudioLoader(parameters.icbhi_root, default_get_filenames))
    if datasets["Ant"]: 
        ant_loader = IcbhiAudioLoader(parameters.ant_root, default_get_filenames)
        ant_loader.name = "Antwerp"
        audio_loaders.append(ant_loader)
        
    # calls the load_all_samples function of each audio loader
    # this returns a dictionary that can access each dataset separately by key, like "Icbhi" or "Jordan"
    audios_dict = load_audios(audio_loaders)

    # from 1 full audio, goes to N chunks depending on slicing parameters
    # still split by dataset, but this time, also by patient!
    audios_c_dict = prepare_audios(audios_dict)

    # split and extend
    # input
    train_audios_c_dict, val_audios_c_dict = split_and_extend(audios_c_dict, parameters.train_test_ratio)

    # val
    val_samples = generate_spec_samples(val_audios_c_dict)
    print(val_samples[0])
    print(val_samples[0][0].shape)

    # train
    train_samples = generate_spec_samples(train_audios_c_dict)
    print(train_samples[0])

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
    
    # TODO: print all folders being used
    print("Tensorflow Version: {}".format(tf.__version__))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    seed_everything()
    arguments = parameters.parse_arguments()
    # print(arguments)
    parameters.init()
    parameters.mode = "pneumonia"
    parameters.file_dir = os.path.join(parameters.cache_root, parameters.mode, os.path.basename(__file__).split('.')[0])
    parameters.description = arguments["description"]
    # TODO: write some of these components into functions
    if int(arguments["testing"]):
        parameters.file_dir += "_testing"
        parameters.n_epochs = 2
        parameters.train_test_ratio = 0.5
        parameters.testing = 1
        parameters.description = "testing"
    initialize_file_folder()
    print("-----------------------")
    # spec_aug_params = []
    # audio_aug_params = []
    # parameters.class_weights = False
    # parameters.shape = (80000, )
    # parameters.n_classes= 2
    # launch_job({"Bd": 0, "Jordan": 1, "Icbhi": 0, "Perch": 0, "Ant": 0, "SimAnt": 0,}, mixednet, spec_aug_params, audio_aug_params, spec_parser)
    parameters.hop_length = 254
    parameters.shape = (128, 311)
    parameters.n_sequences = 9
    spec_aug_params = []
    audio_aug_params = []
    # parameters: dictionary of datasets to use, model  imported from modules/models.py, augmentation parameters, parse function (inside modules/parsers) used to create tf.dataset object in create_tf_dataset
    launch_job({"Bd": 0, "Jordan": 1, "Icbhi": 0, "Perch": 0, "Ant": 0, "SimAnt": 0,}, mixednet, spec_aug_params, audio_aug_params, spec_parser)