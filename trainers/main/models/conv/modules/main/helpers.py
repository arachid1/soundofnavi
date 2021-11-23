from . import parameters
from ..audio_preparer.helpers import return_preparer
from ..augmenter.helpers import return_spec_augmenter, return_audio_augmenter
from ..spec_generator.SpecGenerator import SpecGenerator
from ..spec_generator.SpecGenerator import SpecGenerator
import os
import shutil
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
import numpy as np
import random
from collections import defaultdict

def initialize_job():
    parameters.job_id += 1
    print("Job id is {}.".format(parameters.job_id))
    parameters.job_dir = os.path.join(parameters.file_dir, str(parameters.job_id))
    os.mkdir(parameters.job_dir)
    os.mkdir(os.path.join(parameters.job_dir, "audio"))
    os.mkdir(os.path.join(parameters.job_dir, "audio_c"))
    os.mkdir(os.path.join(parameters.job_dir, "ind_specs"))
    os.mkdir(os.path.join(parameters.job_dir, "patient_avg_specs"))
    os.mkdir(os.path.join(parameters.job_dir, "gradcam"))
    os.mkdir(os.path.join(parameters.job_dir, "others"))
    # make more folders
    # initialize the wanddb stuff

def initialize_file_folder():
    # maybe delete it too, doing it for now for easy testing
    if os.path.exists(parameters.file_dir):
        shutil.rmtree(parameters.file_dir)
    os.mkdir(parameters.file_dir)

def print_dataset(train_labels, val_labels, original_training_length="0", name="Final"):
    if parameters.mode == "cw":
        none_nb, crackles_nb, wheezes_nb, both_nb = train_labels.count([0, 0]), train_labels.count([1, 0]), train_labels.count([0, 1]), train_labels.count([1, 1]) 
        print("--- {} training dataset went from {} to {} elements, with {} none's, {} crakles, {} wheezes and {} both ---".format(name, 0, len(train_labels), none_nb, crackles_nb, wheezes_nb, both_nb))
        none_nb, crackles_nb, wheezes_nb, both_nb = val_labels.count([0, 0]), val_labels.count([1, 0]), val_labels.count([0, 1]), val_labels.count([1, 1]) 
        print("--- {} Validation dataset contains {} elements, with {} none, {} crackles, {} wheezes and {} both ---".format(name, len(val_labels), none_nb, crackles_nb, wheezes_nb, both_nb, ))
    else:
        train_non_pneumonia_nb, train_pneumonia_nb = train_labels.count(0), train_labels.count(1)
        print("--- {} training dataset went from {} to {} elements, with {} 0's, {} 1's and {} others ---".format(name, 0, len(train_labels), train_non_pneumonia_nb, train_pneumonia_nb, len(train_labels) - train_non_pneumonia_nb - train_pneumonia_nb))
        print("--- {} dataset contains {} elements, with {} 0's and {} 1's ---".format(name, len(val_labels), val_labels.count(0), val_labels.count(1)))

"""
input:  
- Dictionary of data split by dataset {Icbhi: [], Jordan: [], ...} 100%
output: 
- Training  dictionary:   {Icbhi: [], Jordan: [], ...}  ratio %
- Val  dictionary:   {Icbhi: [], Jordan: [], ...}       (1 - ratio) %
"""
def split_and_extend(audios_c_dict, train_test_ratio, random_state=12, kfold=False):
    train_dict = defaultdict(lambda: {})
    val_dict = defaultdict(lambda: {})
    print("--- Samples are being split and flattened. ---")
    for key, key_samples in audios_c_dict.items():
        stratify_labels = []
        for patient in key_samples:
            if len(patient) == 0:
                print("len is 0 here")
                continue
            stratify_labels.append(patient[0][1])
        if kfold:
            kf = KFold(n_splits=5, random_state=random_state, shuffle=True)
            kf.get_n_splits(key_samples)
            for i, (train_indexes, val_indexes) in enumerate(kf.split(key_samples)):
                train_dict[i][key] = [key_samples[ind] for ind in train_indexes]
                val_dict[i][key] = [key_samples[ind] for ind in val_indexes]
        else:
            key_val_samples, key_train_samples = train_test_split(key_samples, test_size=train_test_ratio, stratify=stratify_labels, random_state=random_state)
            key_train_samples, key_train_labels = zip(*[[c, c[1]] for audio_chunks in key_train_samples for c in audio_chunks])
            key_val_samples, key_val_labels = zip(*[[c, c[1]] for audio_chunks in key_val_samples for c in audio_chunks])
            key_train_samples = list(key_train_samples)
            key_val_samples = list(key_val_samples)
            train_dict[key] = key_train_samples
            val_dict[key] = key_val_samples
            print_dataset(key_train_labels, key_val_labels, name=key)
    return train_dict, val_dict

def return_components(samples):
    specs = []
    labels = []
    filenames = []
    for s in samples:
        specs.append(s[0])
        labels.append(s[1])
        filenames.append(s[2])
    return specs, labels, filenames

def create_tf_dataset(samples, batch_size, shuffle=False, parse_func=None):
    specs, labels, filenames = return_components(samples)
    dataset = tf.data.Dataset.from_tensor_slices((specs, labels))
    if shuffle: 
        dataset = dataset.shuffle(len(samples))
    if parse_func is not None:
        dataset = dataset.map(lambda spec, label: parse_func(spec, label, parameters.shape, parameters.initial_channels, parameters.cuberooting, parameters.normalizing), num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset, specs, labels, filenames

def seed_everything():
    os.environ["PYTHONHASHSEED"] = "0"
    np.random.seed(0)
    random.seed(0)
    tf.random.set_seed(0)

def load_audios(audio_loaders):
    audios_dict = {}
    for loader in audio_loaders:
        print("Loading {}".format(loader.name))
        loader.load_all_samples()
        audios_dict[loader.name] = loader.return_all_samples()
        print("{} {} audios have been loaded.".format(len(audios_dict[loader.name]), loader.name))
    return audios_dict

def prepare_audios(audios_dict):
    audios_c_dict = {}
    for key, dict_samples in audios_dict.items():
        preparer = return_preparer(key, dict_samples) 
        print("Preparing {}".format(preparer.name))
        preparer.prepare_all_samples()
        # audios_c_dict[preparer.name] = preparer.return_all_samples()
        audios_c_dict[preparer.name] = preparer.return_all_samples_by_patient()
        print("{} {} groups of audio chunks (by filename or patients) have been prepared.".format(len(audios_c_dict[preparer.name]), preparer.name))
    return audios_c_dict

def spec_augment_samples(train_samples, spec_aug_params):

    spec_aug_samples = []
    for params in spec_aug_params:
        augmenter = return_spec_augmenter(train_samples, params[0], params[1])
        augmenter.augment_all_samples()
        augmenter_samples = augmenter.return_all_samples()
        spec_aug_samples.extend(augmenter_samples)
        print("- {} with {} elements with the following params: {} -".format(augmenter.name, len(augmenter_samples), params[1]))
    return spec_aug_samples

def audio_augment_samples(train_audios_c_dict, train_samples, audio_aug_params):
    audio_aug_samples = [] #TODO: Use a dictionary so you can avoid converting augmix to specs
    for key, samples in train_audios_c_dict.items():
        for params in audio_aug_params:
            augmenter = return_audio_augmenter(samples, params[0], params[1])
            augmenter.augment_all_samples()
            augmenter_samples = augmenter.return_all_samples()
            audio_aug_samples.extend(augmenter_samples)
            print("- {} on {} with {} elements with the following params: {} -".format(augmenter.name, key, len(augmenter_samples), params[1]))

    return audio_aug_samples

def generate_spec_samples(audios_c_dict):
    samples = []
    for key, dict_samples in audios_c_dict.items():
        generator = SpecGenerator(dict_samples, key)
        generator.generate_all_samples()
        samples.extend(generator.return_all_samples())
    return samples

def generate_audio_samples(audios_c_dict):
    samples = []
    for key, dict_samples in audios_c_dict.items():
        samples.extend(dict_samples)
    return samples

def set_up_training_samples(train_audios_c_dict, spec_aug_params, audio_aug_params):

    # 1) spectrogram generation
    train_samples = generate_samples(train_audios_c_dict)
    original_training_length = len(train_samples)
    print("--- Original training dataset contains {} elements ---".format(len(train_samples)))

    # 2) spectrogram augmentation 
    spec_aug_samples = spec_augment_samples(train_samples, spec_aug_params)

    # 2) audio augmentation 
    audio_aug_samples = audio_augment_samples(train_audios_c_dict, train_samples, audio_aug_params)

    # concatenating all the train elements
    train_samples.extend(spec_aug_samples)
    train_samples.extend(audio_aug_samples)

    return train_samples, original_training_length


########################

# def custom_split_and_extend(audios_c_dict, train_test_ratio, random_state):
#     train_dict = {}
#     val_dict = {}
#     print("--- Samples are being split and flattened. ---")
#     for key, key_samples in audios_c_dict.items():
#         # print(key_samples[:2])
#         stratify_labels = []
#         for s in key_samples:
#             if len(s) == 0:
#                 # print(s)
#                 # key_samples.remove(s)
#                 print("len is 0 here")
#                 continue
#             stratify_labels.append(s[0][1])
#                 # print(key)
#                 # print("len is 0")
#                 # print(s)
#                 # exit()
#         # stratify_labels = [s[0][1] for s in key_samples]
#         key_val_samples, key_train_samples = train_test_split(key_samples, test_size=train_test_ratio, stratify=stratify_labels, random_state=random_state)
#         # print(key_val_samples[:2])
#         key_train_samples, key_train_labels = zip(*[[c, c[1]] for audio_chunks in key_train_samples for c in audio_chunks])
#         key_val_samples, key_val_labels = zip(*[[c, c[1]] for audio_chunks in key_val_samples for c in audio_chunks])
#         key_train_samples = list(key_train_samples)
#         key_val_samples = list(key_val_samples)
#         # print(type(key_train_samples))
#         # print(type(key_val_samples))
#         train_dict[key] = key_train_samples
#         val_dict[key] = key_val_samples
#         print("- {} has {} training samples with {} 0's and {} 1's, and ".format(key, len(key_train_labels), key_train_labels.count(0), key_train_labels.count(1)), end="")
#         print("{} val samples with {} 0's and {} 1's - ".format(len(key_val_labels), key_val_labels.count(0), key_val_labels.count(1)))
#     return train_dict, val_dict