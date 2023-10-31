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

from modules.models.leaf_model9_model import leaf_model9_model
from modules.models.leaf_model9_model_bis import leaf_model9_model_bis
from modules.models.leaf_model9_model_efnet1 import leaf_model9_model_efnet1
from modules.models.leaf_mixednet_model import leaf_mixednet_model
from modules.models.core import *

from modules.parse_functions.parse_functions import spec_parser, generate_timed_spec

from modules.augmenter.functions import stretch_time, shift_pitch 
from modules.models.core import Distiller

from collections import Counter
from sklearn.utils import class_weight
import tensorflow as tf
import tensorflow_addons as tfa
import leaf_audio.frontend as frontend
from leaf_audio.impulse_responses import gabor_filters
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.cm as cm
import scipy
import imageio
import glob


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

    return train_samples 


def visualize(data, model_to_be_trained, weights_path, epoch_number, _preemp=False):
    

    # weights
    weights = None

    gpus = tf.config.experimental.list_logical_devices('GPU')
    print(gpus)
    cpus = tf.config.experimental.list_logical_devices('CPU')
    print(cpus)

    ################################ NON-LOADED MODEL
    teacher = model_to_be_trained(num_outputs=parameters.n_classes, 
    _frontend=parameters.frontend, encoder=None)
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


    folder = "{}/others".format(parameters.job_dir)
    befores = []
    for i, sample in enumerate(data):
        audio = sample[0]
        audio = np.expand_dims(audio, axis=0)
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)
        with tf.device("cpu:0"):
            before = teacher(audio, False, True)

            # exp3: same as 2 but more advanced
            # layer = teacher._frontend.return_p()
            # outputs = audio[:, :, tf.newaxis] if audio.shape.ndims < 3 else audio
            # fr = layer(np.expand_dims(outputs, axis=0))
            # fr = fr.numpy()
            # visualize_spec_bis(np.swapaxes(np.squeeze(fr), 0, 1), sr=parameters.sr, dest="ir")
            # layer2 = teacher._frontend._activation
            # fr2 = layer(np.expand_dims(outputs, axis=0))
            # tf.print(tf.shape(fr2))
            # fr2 = layer2(tf.expand_dims(fr2, axis=0))
            # fr2 = fr2.numpy()
            # visualize_spec_bis(np.swapaxes(np.squeeze(fr2), 0, 1), sr=parameters.sr, dest="ir_squaredmod")
            # exit()

            # k = teacher._frontend.return_p()._kernel
            # filters = gabor_filters(k, 25).numpy()
            # frq_ranges = np.linspace(0,math.pi,25)
            # frq_ranges = frq_ranges / math.pi
            # frq_ranges = frq_ranges / 2
            # for i, f in enumerate(filters):
            #     # exp 1: plotting frq responses
            #     plt.plot(np.squeeze(audio))
            #     f = np.convolve(f, np.squeeze(audio))
            #     plt.plot(f)
            #     plt.savefig("af_{}".format(i))
            #     plt.close()
            #     if i == 10:
            #         exit()

            # exp 2: plotting the filter responses 
            # plt.plot(frq_ranges, f, 'b')

            # exp4 
        

        before = np.swapaxes(np.squeeze(before.numpy()), 0, 1)
        visualize_spec_bis(before, sr=parameters.sr, dest="{}/before_{}".format(folder, i), title="label_{}_name_{}".format(sample[1], sample[2]))
        befores.append(before)

    kernel_before = teacher._frontend.return_p()._kernel
    weights_before = teacher._frontend.weights
    print("weights before")
    print(weights_before)

    before_filters_centers = weights_before[0+parameters.offset].numpy()[:,0]
    before_filters_centers = before_filters_centers / np.max(before_filters_centers)
    before_filters_centers = before_filters_centers * 2 - 1
    before_filters_centers = before_filters_centers / 2

    before_filters_bdwiths = weights_before[0+parameters.offset].numpy()[:,1]
    before_filters_bdwiths = before_filters_bdwiths / np.max(before_filters_bdwiths)
    before_filters_bdwiths = before_filters_bdwiths * 2 - 1
    before_filters_bdwiths = before_filters_bdwiths / 2

    visualize_filters(before_filters_centers, before_filters_bdwiths, "_before", folder)

    if parameters.leaf:
        before_lowpass_bdwithds = weights_before[1+parameters.offset].numpy()
        before_lowpass_bdwithds = np.squeeze(before_lowpass_bdwithds)
        visualize_filters(np.zeros_like(before_lowpass_bdwithds), before_lowpass_bdwithds, "_lowpass_before", folder)

    ############################ LOADED MODEL
    teacher = model_to_be_trained(num_outputs=parameters.n_classes, 
    _frontend=parameters.frontend, encoder=None)
    teacher.build(shape)

    teacher._load(weights_path, epoch_number)
    
    # teacher._frontend.load_weights(frontend_weights_path)
    # teacher._functional.load_weights(weights_path)
    teacher.compile(
        optimizer=opt,
        loss=loss,
        metrics=[
            'accuracy', 
        ],
    )

    weights_after = teacher._frontend.weights
    print("weights after")
    print(weights_after)
    kernel_after = teacher._frontend.return_p()._kernel

    after_filters_centers = weights_after[0+parameters.offset].numpy()[:,0]
    after_filters_centers = after_filters_centers / np.max(after_filters_centers)
    after_filters_centers = after_filters_centers * 2 - 1
    after_filters_centers = after_filters_centers / 2

    print("filter centers diff")
    print(after_filters_centers - before_filters_centers)
    print(np.average(after_filters_centers - before_filters_centers))

    after_filters_bdwiths = weights_after[0+parameters.offset].numpy()[:,1]
    after_filters_bdwiths = after_filters_bdwiths / np.max(after_filters_bdwiths)
    after_filters_bdwiths = after_filters_bdwiths * 2 - 1
    after_filters_bdwiths = after_filters_bdwiths / 2

    print("filter bdwiths diff")
    print(after_filters_bdwiths - before_filters_bdwiths)
    print(np.average(after_filters_bdwiths - before_filters_bdwiths))

    visualize_filters(after_filters_centers, after_filters_bdwiths, "_after", folder)
    visualize_filters(after_filters_centers - before_filters_centers, after_filters_bdwiths - before_filters_bdwiths, "_diff", folder)

    if parameters.leaf:
        after_lowpass_bdwithds = weights_after[1+parameters.offset].numpy()
        after_lowpass_bdwithds = np.squeeze(after_lowpass_bdwithds)
        print("lowpass filter bdwidths diff")
        print(after_lowpass_bdwithds - before_lowpass_bdwithds)
        print(np.average(after_lowpass_bdwithds - before_lowpass_bdwithds))
        visualize_filters(np.zeros_like(after_lowpass_bdwithds), after_lowpass_bdwithds, "_lowpass_after", folder)
        visualize_filters(np.zeros_like(after_filters_centers), after_lowpass_bdwithds - before_lowpass_bdwithds, "_diff_lowpass", folder)


    # visualize_filters(before_filters_centers, before_filters_bdwiths, "before")
    # visualize_filters(np.zeros_like(lowpass_bdwithds), lowpass_bdwithds, "lowpass_before")

    
    # print(after_filters_centers - before_filters_centers)
    # print(np.average(after_filters_centers))
    # print(np.average(before_filters_centers))
    # print(after_filters_bdwiths - before_filters_bdwiths)
    # print(np.average(after_filters_bdwiths))
    # print(np.average(before_filters_bdwiths))
    # print(weights_before)
    # print(weights_after)
    # print(np.array(weights_after)-np.array(weights_before))
    # print("space")
    # print(kernel_before)
    # print(kernel_after)
    # print(kernel_after-kernel_before)
    # exit()

    for i, sample in enumerate(data):
        audio = sample[0]
        audio = np.expand_dims(audio, axis=0)
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)
        with tf.device("cpu:0"):
            after = teacher(audio, False, True)
        after = np.swapaxes(np.squeeze(after.numpy()), 0, 1)
        visualize_spec_bis(after, sr=parameters.sr, dest="{}/after_{}".format(folder, i), title="label_{}_name_{}".format(sample[1], sample[2]))
        diff = after-befores[i]
        # diff = diff - np.min(diff)
        # diff = diff / np.max(diff)
        # diff = 2*diff - 1
        # print(diff)
        # print(np.max(diff))
        # if np.max(diff)
        # diff = diff / np.max(diff)
        # diff = 2*diff - 1
        visualize_spec_bis(diff, sr=parameters.sr, dest="{}/diff_{}".format(folder, i), title="label_{}_name_{}".format(sample[1], sample[2]))


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def visualize_filters(centers, bandwidths, title, folder):

    colors = cm.winter(np.linspace(0, 1, 80))

    # plt.figure()
    # ar = np.arange(80) # just as an example array
    # for center, color in zip(centers, colors):
    #     plt.plot(center, 0, 'x', c=color)
    # plt.savefig("{}/{}_centers".format(folder, title))
    # plt.close()

    plt.figure()
    ar = np.arange(80) # just as an example array
    for i, (center, color) in enumerate(zip(centers, colors)):
        plt.plot(i, center, 'x', c=color)
    plt.savefig("{}/{}_centers".format(folder, title))
    plt.close()

    # plt.figure()
    # ar = np.arange(80) # just as an example array
    # for bandwidth, color in zip(bandwidths, colors):
    #     plt.plot(bandwidth, 0, 'x', c=color)
    # plt.savefig("{}/{}_bdws".format(folder, title))
    # plt.close()

    plt.figure()
    ar = np.arange(80) # just as an example array
    for i, (bandwidth, color) in enumerate(zip(bandwidths, colors)):
        plt.plot(i, bandwidth, 'x', c=color)
    plt.savefig("{}/{}_bdws".format(folder, title))
    plt.close()

    plt.figure(figsize=(100, 30))
    x_values = np.arange(-2, 2, 0.05)
    # alphas = np.arange(0,1,1/len(centers))
    if title == "_lowpass_after":
        plt.plot(x_values, gaussian(x_values, 0, 0.4), color="green", linestyle="--", linewidth=3)

    for center, bandwidth, color in zip(centers, bandwidths, colors):
        plt.plot(x_values, gaussian(x_values, center, bandwidth), color=color)
    plt.savefig("{}/{}_filters".format(folder, title))
    plt.close()


# def visualize_filters(centers, bandwidths, title, folder):
#     # filters = filters.numpy()[:,0]
#     # print(np.average(filters))
#     # print(filters)
#     # filters = filters.numpy()
#     colors = cm.winter(np.linspace(0, 1, 80))

#     # plt.figure()
#     # ar = np.arange(80) # just as an example array
#     # plt.plot(centers, np.zeros_like(ar) + 0, 'x', c=colors[0])
#     # plt.savefig("{}/{}_centers".format(folder, title))
#     # plt.close()

#     plt.figure()
#     ar = np.arange(80) # just as an example array
#     for center, color in zip(centers, colors):
#         plt.plot(center, 0, 'x', c=color)
#     plt.savefig("{}/{}_centers".format(folder, title))
#     plt.close()

#     # plt.figure()
#     # ar = np.arange(80) # just as an example array
#     # plt.plot(bandwidths, np.zeros_like(ar) + 0, 'x')
#     # plt.savefig("{}/{}_bdws".format(folder, title))
#     # plt.close()

#     plt.figure()
#     ar = np.arange(80) # just as an example array
#     for bandwidth, color in zip(bandwidths, colors):
#         plt.plot(bandwidth, 0, 'x', c=color)
#     plt.savefig("{}/{}_bdws".format(folder, title))
#     plt.close()



#     # plt.figure(figsize=(100, 30))
#     # x_values = np.arange(-2, 2, 0.05)
#     # alphas = np.arange(0,1,1/len(centers))
#     # if title == "_lowpass_after":
#     #     plt.plot(x_values, gaussian(x_values, 0, 0.4), color="green", linestyle="--", linewidth=3)

#     plt.figure(figsize=(100, 30))
#     x_values = np.arange(-2, 2, 0.05)
#     # alphas = np.arange(0,1,1/len(centers))
#     if title == "_lowpass_after":
#         # for x_value, color in zip(x_values, colors):
#         #     plt.plot(x_value, gaussian(x_value, 0, 0.4), color=color, linestyle="--", linewidth=3)
#         plt.plot(x_values, gaussian(x_values, 0, 0.4), color="green", linestyle="--", linewidth=3)

#     # for i in range(len(centers)):
#     #     plt.plot(x_values, gaussian(x_values, centers[i], bandwidths[i]), color="black")
#     for center, bandwidth, color in zip(centers, bandwidths, colors):
#         plt.plot(x_values, gaussian(x_values, center, bandwidth), color=color)
#     plt.savefig("{}/{}_filters".format(folder, title))
#     plt.close()

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

    parameters.n_epochs = 40

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
    parameters.n_classes = 1
    spec_aug_params = []
    audio_aug_params = []
    

    

    samples = train_model({"Bd": 0, "Jordan": 0, "Icbhi": 1, "Perch": 0, "Ant": 0, "SimAnt": 1,}, leaf_model9_model_efnet1, spec_aug_params, audio_aug_params, None, [1,0])
    data = samples[:20]

    parameters.sr ==16000
    parameters.n_filters=80
    parameters.shape = (80,500,3)
    parameters.code = -1
    parameters.stacking = False
    parameters.normalize = False
    parameters.n_classes = 2
    parameters.activation = "sigmoid"
    parameters.model = "effnet"
    weights_path = "/home/alirachidi/classification_algorithm/cache/cw/train196/2/" 
    epoch_number = 2
    parameters.leaf = True
    parameters.frontend = frontend.Leaf(sample_rate=16000, n_filters=80, preemp=True)
    parameters.offset = 1
    initialize_job()
    visualize(data, leaf_model9_model_efnet1, weights_path, epoch_number)

    parameters.sr ==16000
    parameters.n_filters=80
    parameters.shape = (80,500,3)
    parameters.code = -1
    parameters.stacking = False
    parameters.normalize = False
    parameters.n_classes = 2
    parameters.activation = "sigmoid"
    parameters.model = "effnet"
    weights_path = "/home/alirachidi/classification_algorithm/cache/cw/train196/2/" 
    epoch_number = 14
    parameters.leaf = True
    parameters.frontend = frontend.Leaf(sample_rate=16000, n_filters=80, preemp=True)
    parameters.offset = 1
    initialize_job()
    visualize(data, leaf_model9_model_efnet1, weights_path, epoch_number)


    # parameters.sr ==16000
    # parameters.n_filters=80
    # parameters.shape = (80,500,3)
    # parameters.code = -1
    # parameters.stacking = False
    # parameters.normalize = False
    # parameters.n_classes = 2
    # parameters.activation = "sigmoid"
    # weights_path = "/home/alirachidi/classification_algorithm/cache/cw/train192/2/" 
    # epoch_number = 2
    # parameters.leaf = True
    # parameters.frontend = frontend.Leaf(sample_rate=16000, n_filters=80)
    # initialize_job()
    # visualize(data, leaf_model9_model_efnet1, weights_path, epoch_number)

    # parameters.sr ==16000
    # parameters.n_filters=80
    # parameters.shape = (80,500,3)
    # parameters.code = -1
    # parameters.stacking = False
    # parameters.normalize = False
    # parameters.n_classes = 2
    # parameters.activation = "sigmoid"
    # weights_path = "/home/alirachidi/classification_algorithm/cache/cw/train193/2/" 
    # epoch_number = 2
    # parameters.leaf = False
    # parameters.frontend = frontend.SincNet(sample_rate=16000, n_filters=80)
    # initialize_job()
    # visualize(data, leaf_model9_model_efnet1, weights_path, epoch_number)

    # weights_path = "/home/alirachidi/classification_algorithm/cache/cw/train63/1/" 
    # epoch_number = 7
    # initialize_job()
    # visualize(data, leaf_model9_model, weights_path, epoch_number)

    # weights_path = "/home/alirachidi/classification_algorithm/cache/cw/train65/1/" 
    # epoch_number = 12
    # initialize_job()
    # visualize(data, leaf_model9_model, weights_path, epoch_number)

    # weights_path = "/home/alirachidi/classification_algorithm/cache/cw/train66/1/" 
    # epoch_number = 17
    # initialize_job()
    # visualize(data, leaf_model9_model, weights_path, epoch_number)

    # weights_path = "/home/alirachidi/classification_algorithm/cache/cw/train67/1/" 
    # epoch_number = 19
    # initialize_job()
    # visualize(data, leaf_model9_model, weights_path, epoch_number, _preemp=True)

    # weights_path = "/home/alirachidi/classification_algorithm/cache/cw/train68/1/" 
    # epoch_number = 24
    # initialize_job()
    # visualize(data, leaf_model9_model, weights_path, epoch_number, _preemp=True)

    # weights_path = "/home/alirachidi/classification_algorithm/cache/cw/train69/1/" 
    # epoch_number = 24
    # initialize_job()
    # visualize(data, leaf_model9_model, weights_path, epoch_number)

    # weights_path = "/home/alirachidi/classification_algorithm/cache/cw/train70/1/" 
    # epoch_number = 24
    # initialize_job()
    # visualize(data, leaf_model9_model, weights_path, epoch_number)

    # weights_path = "/home/alirachidi/classification_algorithm/cache/cw/train71/1/" 
    # epoch_number = 13
    # initialize_job()
    # visualize(data, leaf_model9_model, weights_path, epoch_number)

    # parameters.n_classes = 2

    # weights_path = "/home/alirachidi/classification_algorithm/cache/cw/train72/1/" 
    # epoch_number = 16
    # initialize_job()
    # visualize(data, leaf_model9_model, weights_path, epoch_number)

    # descriptions = ["crackles only", "crackles only with mixed up data",  "crackles only with mixed up data and pre-emphasize layer", 
    #     "crackles only with mixed up data and pre-emphasize layer 2.0", "crackles only with mixed up data and pre-emphasize layer 2.0 and different loss",
    #     "crackles only with mixed up data and pre-emphasize layer 2.0 and different loss 2.0"]

    # # spectrograms
    # fig, axes = plt.subplots(7, 1,figsize=(30,30)) 

    # for i in range(1, 7):
    #     image_path = "/home/alirachidi/classification_algorithm/cache/cw/train48_testing/{}/others/after_10__label_[1.0, 0.0]_name_226_1b1_Al_sc_Meditron_0.png".format(i)
    #     image = imageio.imread(image_path)
    #     ax = axes[i-1]
    #     ax.set_title(descriptions[i-1])
    #     ax.axis('off')
    #     ax.imshow(image)
    # plt.savefig("/home/alirachidi/classification_algorithm/cache/cw/train48_testing/spectrograms.png")


    # # filters centers
    # for i in range(1, 7):
    #     image_path = "/home/alirachidi/classification_algorithm/cache/cw/train48_testing/{}/others/_diff_centers.png".format(i)
    #     image = imageio.imread(image_path)
    #     ax = axes[i-1]
    #     ax.set_title(descriptions[i-1])
    #     ax.axis('off')
    #     ax.imshow(image)
    # plt.savefig("/home/alirachidi/classification_algorithm/cache/cw/train48_testing/diff_centers.png")

    # # filters bdwiths
    # for i in range(1, 7):
    #     image_path = "/home/alirachidi/classification_algorithm/cache/cw/train48_testing/{}/others/_diff_bdws.png".format(i)
    #     image = imageio.imread(image_path)
    #     ax = axes[i-1]
    #     ax.set_title(descriptions[i-1])
    #     ax.axis('off')
    #     ax.imshow(image)
    # plt.savefig("/home/alirachidi/classification_algorithm/cache/cw/train48_testing/diff_bdwidths.png")

    # # lowpass bandwidths
    # for i in range(1, 7):
    #     image_path = "/home/alirachidi/classification_algorithm/cache/cw/train48_testing/{}/others/_diff_lowpass_bdws.png".format(i)
    #     image = imageio.imread(image_path)
    #     ax = axes[i-1]
    #     ax.set_title(descriptions[i-1])
    #     ax.axis('off')
    #     ax.imshow(image)
    # plt.savefig("/home/alirachidi/classification_algorithm/cache/cw/train48_testing/diff_lowpass.png")


    # s, s_copy = launch_job({"Bd": 0, "Jordan": 0, "Icbhi": 1, "Perch": 1, "Ant": 1, "SimAnt": 1,}, leaf_mixednet_model, spec_aug_params, audio_aug_params, None, [1,0], weights_path)
    # parameters.mode="cw"
    # weights_path = "/home/alirachidi/classification_algorithm/cache/cw/train45/2/teacher_40"
    # _, _ = launch_job({"Bd": 0, "Jordan": 0, "Icbhi": 1, "Perch": 1, "Ant": 1, "SimAnt": 1,}, leaf_mixednet_model, spec_aug_params, audio_aug_params, None, [0,1], weights_path, s, s_copy)
    
    # to run another job, add a line to modify whatever parameters, and rerun a launch_job function as many times as you want!
    # parameters.hop_length = 512
    # launch_job({"Bd": 0, "Jordan": 1, "Icbhi": 1, "Perch": 0, "Ant": 0, "SimAnt": 0,}, model9, spec_aug_params, audio_aug_params, spec_parser)
