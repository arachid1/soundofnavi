from . import parameters
from ..audio_preparer.helpers import return_preparer
from ..augmenter.helpers import return_spec_augmenter, return_audio_augmenter
from sklearn.model_selection import train_test_split, RepeatedKFold, RepeatedStratifiedKFold
import tensorflow as tf
from collections import defaultdict
import matplotlib.pyplot as plt

dataset_dict = {'Antwerp': 0, 'Icbhi': 1, 'Perch': 2}
sthethoscope_dict = {'Antwerp': 0, 'Meditron': 1,
                     'Litt3200': 2, 'LittC2SE': 3, 'AKGC417L': 4, 'Perch': 5}

# currently dataset and mic; add chest location, gender, age, first/middle/last chunk


def codify(samples):
    codes = []
    for s in samples:
        domains = []
        name = s[2]
        domains.append(name)
        if name.startswith("RESPT"):  # Antwerp
            domains.append(dataset_dict['Antwerp'])
            domains.append(sthethoscope_dict["Antwerp"])
        elif len(name.split('_')[0]) == 3:  # Icbhi
            domains.append(dataset_dict['Icbhi'])
            elements = name.split('_')
            if elements[-2] == "Litt3200":
                print(elements)
            domains.append(sthethoscope_dict[elements[-2]])
        elif len(name.split('-')[0]) == 6:  # Perch
            domains.append(dataset_dict["Perch"])
            domains.append(sthethoscope_dict['Perch'])
        assert (len(domains) == 3), "domains is empty"
        # s[2] = domains
        codes.append(domains)
    return codes


def print_dataset(train_labels, val_labels, original_training_length="0", name="Final"):
    if parameters.mode == "cw":
        none_nb, crackles_nb, wheezes_nb, both_nb = train_labels.count([0, 0]), train_labels.count(
            [1, 0]), train_labels.count([0, 1]), train_labels.count([1, 1])
        print("--- {} training dataset went from {} to {} elements, with {} none's, {} crakles, {} wheezes and {} both ---".format(
            name, 0, len(train_labels), none_nb, crackles_nb, wheezes_nb, both_nb))
        none_nb, crackles_nb, wheezes_nb, both_nb = val_labels.count(
            [0, 0]), val_labels.count([1, 0]), val_labels.count([0, 1]), val_labels.count([1, 1])
        print("--- {} Validation dataset contains {} elements, with {} none, {} crackles, {} wheezes and {} both ---".format(
            name, len(val_labels), none_nb, crackles_nb, wheezes_nb, both_nb, ))
    else:
        train_non_pneumonia_nb, train_pneumonia_nb = train_labels.count(
            0), train_labels.count(1)
        print("--- {} training dataset went from {} to {} elements, with {} 0's, {} 1's and {} others ---".format(name, 0,
                                                                                                                  len(train_labels), train_non_pneumonia_nb, train_pneumonia_nb, len(train_labels) - train_non_pneumonia_nb - train_pneumonia_nb))
        print("--- {} Validation dataset contains {} elements, with {} 0's and {} 1's ---".format(
            name, len(val_labels), val_labels.count(0), val_labels.count(1)))


"""
2 main functionalities:
splits the data by patient
extends and rearranges such that it's splits by dataset + train vs val
"""


def split_and_extend(audios_c_dict, train_test_ratio, random_state=12, kfold=False):

    train_dict = defaultdict(lambda: {})
    val_dict = defaultdict(lambda: {})

    print("--- Samples are being split into training/val groups by patient by dataset ---")
    for dataset_id, dataset_samples in audios_c_dict.items():
        samples_by_patient_by_dataset = []
        for patient_id, patient_samples in dataset_samples.items():
            samples_by_patient_by_dataset.append(patient_samples)
        kf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=random_state)
        kf.get_n_splits(samples_by_patient_by_dataset)
        for i, (train_indexes, val_indexes) in enumerate(kf.split(samples_by_patient_by_dataset)):
            train_dict[i][dataset_id] = [samples_by_patient_by_dataset[ind]
                                      for ind in train_indexes]
            val_dict[i][dataset_id] = [samples_by_patient_by_dataset[ind] for ind in val_indexes]
            # print(train_dict[i][dataset_id][0])
            # print(val_dict[i][dataset_id][0])
            # exit()
    # for key, key_samples in audios_c_dict.items():
    #     stratify_labels = []
    #     # collecting PATIENT labels for PNEUMONIA <- this wouldn't be the same C/W or NON-PATIENT labelling
    #     for patient in key_samples:
    #         print(patient)
    #         if len(patient) == 0:
    #             print("Empty patient while splitting")
    #             continue
    #         stratify_labels.append(patient[0][1])
    #     if kfold:
    #         # kf = KFold(n_splits=10, random_state=random_state, shuffle=True)
    #         kf = RepeatedKFold(n_splits=5, n_repeats=2,
    #                            random_state=random_state)
    #         kf.get_n_splits(key_samples)
    #         for i, (train_indexes, val_indexes) in enumerate(kf.split(key_samples)):
    #             train_dict[i][key] = [key_samples[ind]
    #                                   for ind in train_indexes]
    #             train_dict[i][key] = [
    #                 item for patientwise in train_dict[i][key] for item in patientwise]
    #             val_dict[i][key] = [key_samples[ind] for ind in val_indexes]
    #             val_dict[i][key] = [item for patientwise in val_dict[i][key]
    #                                 for item in patientwise]
    #     # else:
        #     try:
        #         key_val_samples, key_train_samples = train_test_split(
        #             key_samples, test_size=train_test_ratio, stratify=stratify_labels, random_state=random_state)
        #     except ValueError:
        #         key_val_samples, key_train_samples = train_test_split(
        #             key_samples, test_size=train_test_ratio, random_state=random_state)

        #     # flattening from grouped by patient to simple audios (because the split has already been done so we don't care if it's organize by patient!)
        #     key_train_samples, key_train_labels = zip(
        #         *[[c, c[1]] for audio_chunks in key_train_samples for c in audio_chunks])
        #     key_val_samples, key_val_labels = zip(
        #         *[[c, c[1]] for audio_chunks in key_val_samples for c in audio_chunks])

        #     key_train_samples = list(key_train_samples)
        #     key_val_samples = list(key_val_samples)
        #     train_dict[key] = key_train_samples
        #     val_dict[key] = key_val_samples
        #     print_dataset(key_train_labels, key_val_labels, name=key)
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
        dataset = dataset.map(lambda spec, label: parse_func(spec, label, parameters.shape,
                                                             parameters.initial_channels, parameters.cuberooting, parameters.normalizing), num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset, specs, labels, filenames


def load_audios(audio_loaders):
    audios_dict = {}
    for loader in audio_loaders:
        print("- Loading {}.".format(loader.name))
        loader.load_all_samples()
        audios_dict[loader.name] = loader.return_all_samples()
        print("{} {} audios have been loaded.".format(
            len(audios_dict[loader.name]), loader.name))
    return audios_dict


def prepare_audios(audios_dict):
    audios_c_dict = {}
    for key, dict_samples in audios_dict.items():
        preparer = return_preparer(key, dict_samples)
        print("- Preparing {}.".format(preparer.name))
        preparer.prepare_all_samples()
        # audios_c_dict[preparer.name] = preparer.return_all_samples()
        audios_c_dict[preparer.name] = preparer.return_all_samples_by_patient()
        print("{} {} groups of audio chunks (by filename or patients) have been prepared.".format(
            len(audios_c_dict[preparer.name]), preparer.name))
    return audios_c_dict


def spec_augment_samples(train_samples, spec_aug_params):

    spec_aug_samples = []
    for params in spec_aug_params:
        print("- Augmenting spectrograms.")
        augmenter = return_spec_augmenter(train_samples, params[0], params[1])
        augmenter.augment_all_samples()
        augmenter_samples = augmenter.return_all_samples()
        spec_aug_samples.extend(augmenter_samples)
        print("Applied {} with {} elements with the following params: {} -".format(
            augmenter.name, len(augmenter_samples), params[1]))
    return spec_aug_samples


def audio_augment_samples(train_audios_c_dict, train_samples, audio_aug_params):
    # TODO: Use a dictionary so you can avoid converting augmix to specs
    audio_aug_samples = []
    for key, samples in train_audios_c_dict.items():
        print("- Augmenting audios.")
        for params in audio_aug_params:
            augmenter = return_audio_augmenter(samples, params[0], params[1])
            augmenter.augment_all_samples()
            augmenter_samples = augmenter.return_all_samples()
            audio_aug_samples.extend(augmenter_samples)
            print("Applied {} on {} with {} elements with the following params: {} -".format(
                augmenter.name, key, len(augmenter_samples), params[1]))

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
    train_samples = generate_spec_samples(train_audios_c_dict)
    original_training_length = len(train_samples)
    print("--- Original training dataset contains {} elements ---".format(len(train_samples)))

    # 2) spectrogram augmentation
    # cross-dataset (for example, Augmix can use an image from Icbhi and Jordan)
    spec_aug_samples = spec_augment_samples(train_samples, spec_aug_params)

    # 2) audio augmentation
    # NOT cross-dataset
    audio_aug_samples = audio_augment_samples(
        train_audios_c_dict, train_samples, audio_aug_params)

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


def oversample(_train_samples):
    none_samples = [s for s in _train_samples if s[1] == [1, 0, 0, 0]]
    crackles_samples = [s for s in _train_samples if s[1] == [0, 1, 0, 0]]
    wheezes_samples = [s for s in _train_samples if s[1] == [0, 0, 1, 0]]
    both_samples = [s for s in _train_samples if s[1] == [0, 0, 0, 1]]

    print(len(none_samples))
    print(len(crackles_samples))
    print(len(wheezes_samples))
    print(len(both_samples))

    print("lengths")
    print(len(_train_samples))
    print(len(_val_samples))

    original_length = len(crackles_samples)
    i = 0
    while(len(crackles_samples) < len(none_samples)):
        index = i % original_length
        crackles_samples.append(crackles_samples[index])
        i += 1

    original_length = len(wheezes_samples)
    i = 0
    while(len(wheezes_samples) < len(none_samples)):
        index = i % original_length
        wheezes_samples.append(wheezes_samples[index])
        i += 1

    original_length = len(both_samples)
    i = 0
    while(len(both_samples) < len(none_samples)):
        index = i % original_length
        both_samples.append(both_samples[index])
        i += 1

    print(len(none_samples))
    print(len(crackles_samples))
    print(len(wheezes_samples))
    print(len(both_samples))

    _train_samples = none_samples + crackles_samples + wheezes_samples + both_samples
    return _train_samples


def return_official_icbhi_split(_all_samples, official_labels_path):
    _all_samples = [item for sublist in _all_samples for item in sublist]
    # print(_all_samples)
    # _all_samples = _train_samples +_val_samples
    print("len of icbhi")
    print(len(_all_samples))

    official_train_samples = []
    official_val_samples = []
    count = 0
    with open(official_labels_path) as file:
        for line in file:
            trigger = True
            el = line.rstrip().split()
            name = el[0]
            category = el[1]
            print(name)
            for s in _all_samples:
                if s[2].startswith(name):
                    if trigger:
                        count += 1
                        trigger = False
                    if category == 'test':
                        official_val_samples.append(s)
                    else:
                        official_train_samples.append(s)

    _train_samples = official_train_samples
    _val_samples = official_val_samples
    print("len of train icbhi")
    print(len(_train_samples))
    print("len of val icbhi")
    print(len(_val_samples))
    return _train_samples, _val_samples
