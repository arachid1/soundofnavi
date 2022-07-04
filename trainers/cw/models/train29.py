import sys
# sys.path.insert(0, 'main/models/conv')
from modules.main import parameters
from modules.main.helpers import *
from modules.main.global_helpers import *

from modules.audio_loader.JordanAudioLoader import JordanAudioLoader
from modules.audio_loader.IcbhiAudioLoader import IcbhiAudioLoader
from modules.audio_loader.BdAudioLoader import BdAudioLoader
from modules.audio_loader.PerchAudioLoader import PerchAudioLoader
from modules.audio_loader.helpers import default_get_filenames, bd_get_filenames, perch_get_filenames

from modules.spec_generator.SpecGenerator import SpecGenerator

from modules.callbacks.NewCallback import NewCallback

from modules.models.leaf_model import leaf_model

from modules.parse_functions.parse_functions import spec_parser, generate_timed_spec

from modules.augmenter.functions import stretch_time, shift_pitch 

from collections import Counter
from sklearn.utils import class_weight
import tensorflow as tf
import tensorflow_addons as tfa
import leaf_audio.frontend as frontend
import numpy as np

def train_model(datasets, model_to_be_trained, spec_aug_params, audio_aug_params, parse_function, window_len, window_stride):

    
    # simply initialize audio loader object for each dataset
    # mandatory parameters:  (1) root of dataset (2) function for extracting filenames 
    # optional parameters: or other custom parameters, like the Bangladesh excel path
    # NOTE: name attribute: to distinguish between datasets when the same audio loader object is used for different datasets, such as antwerp and icbhi that both use IcbhiAudioLoader

    audio_loaders = []
    
    if datasets["Jordan"]: audio_loaders.append(JordanAudioLoader(parameters.jordan_root, default_get_filenames))
    if datasets["Bd"]: audio_loaders.append(BdAudioLoader(parameters.bd_root, bd_get_filenames, parameters.excel_path))
    if datasets["Perch"]: audio_loaders.append(PerchAudioLoader(parameters.perch_root, perch_get_filenames))
    if datasets["Icbhi"]: audio_loaders.append(IcbhiAudioLoader(parameters.icbhi_root, default_get_filenames))
    if datasets["Ant"]: 
        # TODO: pass names?
        ant_loader = IcbhiAudioLoader(parameters.ant_root, default_get_filenames)
        ant_loader.name = "Antwerp"
        audio_loaders.append(ant_loader)
        
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
    # print(audios_c_dict)

    # NOTE: # Data is grouped by dataset and patient thus far
    # this functions (1) splits each dataset into train and validation, then (2) after split, we don't care about grouping by patient = flatten to list of audios by patients to give a list of audios 
    #  input: Full Dictionary:  {Icbhi: [] -> data grouped by PATIENT, Jordan: [] -> data grouped by PATIENT, ...}
    # output: Training /// Val  dictionary:   {Icbhi: [] -> data organized INDIVIDUALLY, Jordan: [] -> data organized  INDIVIDUALLY} 
    train_audios_c_dict, val_audios_c_dict = split_and_extend(audios_c_dict, parameters.train_test_ratio)
    # NOTE: # Data is only grouped by dataset now
    

    # simplest step: now that everything is ready, we convert to spectrograms! it's the most straightforward step...
    # convert: [audio, label, filename] -> [SPEC, label, filename]
    val_samples = generate_audio_samples(val_audios_c_dict)
    # ... but it's different for training because of augmentation. the following function sets up and merges 2 branches:
    #   1) augment AUDIO and convert to spectrogram
    #   2) convert to spectrogram and augment SPECTROGRAM
    # train_samples, original_training_length = set_up_training_samples(train_audios_c_dict, spec_aug_params, audio_aug_params) 
    train_samples = generate_audio_samples(train_audios_c_dict)
    # train_samples = generate_spec_samples(train_audios_c_dict) # the same as above if no augmentation 
     # NOTE: # Data is NOT LONGER grouped by dataset 

    # from now on it's cake!
    train_dataset, __, train_labels, __ = create_tf_dataset(train_samples, batch_size=parameters.batch_size, shuffle=True, parse_func=parse_function)
    val_dataset, val_specs, val_labels, val_filenames = create_tf_dataset(val_samples, batch_size=1, shuffle=False, parse_func=parse_function) # keep shuffle = False!
    train_non_pneumonia_nb, train_pneumonia_nb = train_labels.count(0), train_labels.count(1)
    # print(list(train_dataset.as_numpy_iterator())[0])
    # print(list(train_dataset.as_numpy_iterator())[0][0].shape)
    # exit()
    print("-----------------------")
    print_dataset(train_labels, val_labels)

    # weights
    weights = None

    # if bool(parameters.class_weights):
    #     print("Initializing weights...")
    #     weights = class_weight.compute_class_weight(
    #         "balanced", [0, 1], [l for l in train_labels if l == 0 or l == 1])
    #     weights = {i: weights[i] for i in range(0, len(weights))}
    #     print("weights = {}".format(weights))
    
    # handles metrics, file saving (all the files inside gradcam/, tp/, others/, etc), report writing (report.txt), visualizations, etc
    parameters.adaptive_lr = False
    metrics_callback = NewCallback(val_dataset, val_filenames)

    gpus = tf.config.experimental.list_logical_devices('GPU')

    # model setting
    model = model_to_be_trained(num_outputs=parameters.n_classes, frontend=frontend.Leaf(sample_rate=16000, n_filters=80, window_len=window_len, window_stride=window_stride), encoder=None)
    shape = (None, parameters.audio_length*parameters.sr)
    model.build(shape)

    optimizers = [
        tf.keras.optimizers.Adam(
            lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=parameters.epsilon, decay=parameters.weight_decay, amsgrad=False
        ),
        tf.keras.optimizers.Adam(
            lr=parameters.lr, beta_1=0.9, beta_2=0.999, epsilon=parameters.epsilon, decay=parameters.weight_decay, amsgrad=False
        )
    ]
    optimizers_and_layers = [(optimizers[0], model.layers[0]), (optimizers[1], model.layers[1:])]
    opt = tfa.optimizers.MultiOptimizer(optimizers_and_layers)

    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=parameters.label_smoothing)

    # training backend first
    # for l in model.layers:
    #     if l.name == "leaf":
    #         l.trainable = False
    #     else:
    #         l.trainable = True

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=[
            'accuracy', 
        ],
    )

    model.summary(line_length=110)

    folder = "{}/others".format(parameters.job_dir)
    sps = []
    s = list(train_dataset.as_numpy_iterator())[0][0][:3]
    for i, sp in enumerate(s):
        sp = np.expand_dims(sp, axis=0)
        sp = model(sp, return_spec=True)
        sp = np.swapaxes(np.squeeze(sp.numpy()), 0, 1)
        sps.append(sp)
        visualize_spec_bis(sp, sr=parameters.sr, dest="{}/backend_before_{}".format(folder, i))

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

    for i, sp2 in enumerate(s):
        sp2 = np.expand_dims(sp2, axis=0)
        sp2 = model(sp2, return_spec=True)
        sp2 = np.swapaxes(np.squeeze(sp2.numpy()), 0, 1)
        visualize_spec_bis(sp2, sr=parameters.sr, dest="{}/backend_after_{}".format(folder, i))

        diff = sp2-sps[i]
        visualize_spec_bis(diff, sr=parameters.sr, dest="{}/backend_diff_{}".format(folder, i))
    

    # # training frontend
    # for l in model.layers:
    #     if l.name == "leaf":
    #         l.trainable = True
    #     else:
    #         l.trainable = False

    # # parameters.lr = 1e-4
    # # parameters.n_epochs = 5
    # # opt = tf.keras.optimizers.Adam(
    # #     lr=parameters.lr, beta_1=0.9, beta_2=0.999, epsilon=parameters.epsilon, decay=parameters.weight_decay, amsgrad=False
    # # )
    # optimizers = [
    #     tf.keras.optimizers.Adam(
    #         lr=parameters.lr, beta_1=0.9, beta_2=0.999, epsilon=parameters.epsilon, decay=parameters.weight_decay, amsgrad=False
    #     ),
    #     tf.keras.optimizers.Adam(
    #         lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=parameters.epsilon, decay=parameters.weight_decay, amsgrad=False
    #     )
    # ]
    # optimizers_and_layers = [(optimizers[0], model.layers[0]), (optimizers[1], model.layers[1:])]
    # opt = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    
    # model.compile(
    #     optimizer=opt,
    #     loss=loss,
    #     metrics=[
    #         'accuracy', 
    #     ],
    # )

    # # folder = "{}/others".format(parameters.job_dir)
    # sps = []
    # s = list(train_dataset.as_numpy_iterator())[0][0][:3]
    # for i, sp in enumerate(s):
    #     sp = np.expand_dims(sp, axis=0)
    #     sp = model(sp, return_spec=True)
    #     sp = np.swapaxes(np.squeeze(sp.numpy()), 0, 1)
    #     sps.append(sp)
    #     visualize_spec_bis(sp, sr=parameters.sr, dest="{}/frontend_before_{}".format(folder, i))

    # if len(gpus) > 1:
    #     print("You are using 2 GPUs while the code is set up for one only.")
    #     exit()

    # # training
    # model.fit(
    #     train_dataset,
    #     epochs=parameters.n_epochs,
    #     verbose=2,
    #     class_weight=weights,
    #     callbacks=[metrics_callback]
    # )

    # for i, sp2 in enumerate(s):
    #     sp2 = np.expand_dims(sp2, axis=0)
    #     sp2 = model(sp2, return_spec=True)
    #     sp2 = np.swapaxes(np.squeeze(sp2.numpy()), 0, 1)
    #     visualize_spec_bis(sp2, sr=parameters.sr, dest="{}/frontend_after_{}".format(folder, i))

    #     diff = sp2-sps[i]
    #     visualize_spec_bis(diff, sr=parameters.sr, dest="{}/frontend_diff_{}".format(folder, i))

    # model.save(parameters.job_dir + "/model_{}.h5".format(parameters.n_epochs))

def launch_job(datasets, model, spec_aug_params, audio_aug_params, parse_function, window_len, window_stride):
    '''
    parameters: 
    # dictionary:  datasets to use, 
    # model:  imported from modules/models.py, 
    # augmentation parameters:  (No augmentation if empty list is passed), 
    # parse function:  (inside modules/parsers) used to create tf.dataset object in create_tf_dataset
    '''
    # in a given file named train$ (parent/cache folder named train$), we can have multiple jobs (child folders named 1,2,3)
    initialize_job() #  initialize each (child) job inside the file (i.e, creates all the subfolders like tp/tn/gradcam/etc, file saving conventions, etc)
    train_model(datasets, model, spec_aug_params, audio_aug_params, parse_function, window_len, window_stride)

if __name__ == "__main__":
    
    print("Tensorflow Version: {}".format(tf.__version__))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    seed_everything() # seeding np, tf, etc
    arguments = parameters.parse_arguments()
    parameters.init()
    parameters.mode = "pneumonia" if arguments["mode"] == "main" else arguments["mode"]
    parameters.n_classes = 2 if parameters.mode == "cw" else 1
    print(parameters.cache_root)
    print(parameters.mode)
    print(os.path.basename(__file__).split('.')[0])
    parameters.file_dir = os.path.join(parameters.cache_root, parameters.mode, os.path.basename(__file__).split('.')[0])
    parameters.description = arguments["description"]
    testing_mode(int(arguments["testing"])) # if true, adds _testing to the cache folder, sets a small number of epochs, smaller dataset, turn off a few settings in the callback, etc
    # works to set up the parent folder (i.e., overwrites if already exists) + works well with initialize_job above
    initialize_file_folder()
    print("-----------------------")


    ###### set up used for spec input models (default)
    parameters.hop_length = 254
    parameters.shape = (128, 311)
    parameters.n_sequences = 9
    spec_aug_params = []
    audio_aug_params = []
    for wsz in [50, 100]:
        for wss in [10, 25, 50]:
            launch_job({"Bd": 0, "Jordan": 0, "Icbhi": 1, "Perch": 1, "Ant": 1, "SimAnt": 1,}, leaf_model, spec_aug_params, audio_aug_params, None, wsz, wss)
    # launch_job({"Bd": 0, "Jordan": 0, "Icbhi": 1, "Perch": 1, "Ant": 1, "SimAnt": 1,}, leaf_model, spec_aug_params, audio_aug_params, None)
    
    # to run another job, add a line to modify whatever parameters, and rerun a launch_job function as many times as you want!
    # parameters.hop_length = 512
    # launch_job({"Bd": 0, "Jordan": 1, "Icbhi": 1, "Perch": 0, "Ant": 0, "SimAnt": 0,}, model9, spec_aug_params, audio_aug_params, spec_parser)
