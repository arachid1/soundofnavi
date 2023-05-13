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
from modules.callbacks.visualizationCallback import visualizationCallback

from modules.models.leaf_model9_model import leaf_model9_model
from modules.models.leaf_mixednet_model import leaf_mixednet_model

from modules.parse_functions.parse_functions import spec_parser, generate_timed_spec

from modules.augmenter.functions import stretch_time, shift_pitch 
from modules.models.core import Distiller

from collections import Counter
from sklearn.utils import class_weight
import tensorflow as tf
import tensorflow_addons as tfa
import leaf_audio.frontend as frontend
import numpy as np


class Autoencoder(tf.keras.Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      tf.keras.layers.GlobalAveragePooling2D(),
      tf.keras.layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      tf.keras.layers.Dense(1024, activation='sigmoid'),
      tf.keras.layers.Reshape((32, 32))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

def train_model(datasets, model_to_be_trained, spec_aug_params, audio_aug_params, parse_function, label_passed):

    
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

    # for i in range(len(train_samples)):
    #     if train_samples[i][1] == label_passed or train_samples[i][1] == [1, 1]: 
    #         train_samples[i][1] = 1
    #     else:
    #         train_samples[i][1] = 0

    # for i in range(len(val_samples)):
    #     if val_samples[i][1] == label_passed or val_samples[i][1] == [1, 1]: 
    #         val_samples[i][1] = 1
    #     else:
    #         val_samples[i][1] = 0
    
    # parameters.mode = "main"
    # from now on it's cake!
    train_dataset, __, train_labels, __ = create_tf_dataset(train_samples, batch_size=parameters.batch_size, shuffle=True, parse_func=parse_function)
    val_dataset, val_specs, val_labels, val_filenames = create_tf_dataset(val_samples, batch_size=1, shuffle=False, parse_func=parse_function) # keep shuffle = False!
    train_non_pneumonia_nb, train_pneumonia_nb = train_labels.count(0), train_labels.count(1)
    print(train_non_pneumonia_nb)
    print(train_pneumonia_nb)
    print(val_labels.count(0))
    print(val_labels.count(1))

    samples = val_samples[:10]
    print("-----------------------")
    print_dataset(train_labels, val_labels)

    # weights
    weights = None
    
    # handles metrics, file saving (all the files inside gradcam/, tp/, others/, etc), report writing (report.txt), visualizations, etc
    metrics_callback = NewCallback(val_dataset, val_filenames)
    viz_callback = visualizationCallback(samples)

    gpus = tf.config.experimental.list_logical_devices('GPU')

    # teacher
    teacher = model_to_be_trained(num_outputs=parameters.n_classes, frontend=frontend.Leaf(sample_rate=16000, n_filters=80), encoder=Autoencoder(latent_dim=64))
    shape = (None, parameters.audio_length*parameters.sr)
    teacher.build(shape)

    optimizers = [
        tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10000,
        decay_rate=0.9), beta_1=0.9, beta_2=0.999, epsilon=parameters.epsilon, decay=parameters.weight_decay, amsgrad=False
        ),
        tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=5e-3,
        decay_steps=10000,
        decay_rate=0.9), beta_1=0.9, beta_2=0.999, epsilon=parameters.epsilon, decay=parameters.weight_decay, amsgrad=False
        )
    ]
    optimizers_and_layers = [(optimizers[0], teacher.layers[0]), (optimizers[1], teacher.layers[1:])]
    opt = tfa.optimizers.MultiOptimizer(optimizers_and_layers)

    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=parameters.label_smoothing)

    teacher.compile(
        optimizer=opt,
        loss=loss,
        metrics=[
            'accuracy', 
        ],
    )

    teacher.summary(line_length=110)

    teacher.fit(
        train_dataset,
        epochs=parameters.n_epochs,
        verbose=2,
        class_weight=weights,
        callbacks=[metrics_callback, viz_callback]
    )


    
def launch_job(datasets, model, spec_aug_params, audio_aug_params, parse_function, label_passed):
    '''
    parameters: 
    # dictionary:  datasets to use, 
    # model:  imported from modules/models.py, 
    # augmentation parameters:  (No augmentation if empty list is passed), 
    # parse function:  (inside modules/parsers) used to create tf.dataset object in create_tf_dataset
    '''
    # in a given file named train$ (parent/cache folder named train$), we can have multiple jobs (child folders named 1,2,3)
    initialize_job() #  initialize each (child) job inside the file (i.e, creates all the subfolders like tp/tn/gradcam/etc, file saving conventions, etc)
    train_model(datasets, model, spec_aug_params, audio_aug_params, parse_function, label_passed)

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
    print("Description: {}".format(parameters.description))

    parameters.n_epochs = 25

    testing_mode(int(arguments["testing"])) # if true, adds _testing to the cache folder, sets a small number of epochs, smaller dataset, turn off a few settings in the callback, etc
    # works to set up the parent folder (i.e., overwrites if already exists) + works well with initialize_job above
    initialize_file_folder()
    print("-----------------------")


    ###### set up used for spec input models (default)
    parameters.hop_length = 254
    parameters.shape = (128, 311)
    parameters.n_sequences = 9
    
    parameters.lr_patience = 4
    parameters.es_patience = 10
    # parameters.mode = "main"
    parameters.adaptive_lr = False
    # parameters.mode = "cw"
    parameters.n_classes = 2
    spec_aug_params = []
    audio_aug_params = []
    launch_job({"Bd": 0, "Jordan": 0, "Icbhi": 1, "Perch": 1, "Ant": 1, "SimAnt": 1,}, leaf_mixednet_model, spec_aug_params, audio_aug_params, None, [1,0])

    
    # to run another job, add a line to modify whatever parameters, and rerun a launch_job function as many times as you want!
    # parameters.hop_length = 512
    # launch_job({"Bd": 0, "Jordan": 1, "Icbhi": 1, "Perch": 0, "Ant": 0, "SimAnt": 0,}, model9, spec_aug_params, audio_aug_params, spec_parser)
