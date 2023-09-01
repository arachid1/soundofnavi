import os
import numpy as np
import random
import tensorflow as tf
from matplotlib import pyplot as plt
import librosa
from librosa import display

def seed_everything():
    os.environ["PYTHONHASHSEED"] = "0"
    np.random.seed(0)
    random.seed(0)
    tf.random.set_seed(0)

def print_sample_count(none, crackles, wheezes, both):
    print('all:{}\nnone:{}\ncrackles:{}\nwheezes:{}\nboth:{}'.format(
        len(none) + len(crackles) + len(wheezes) + len(both),
        len(none),
        len(crackles),
        len(wheezes),
        len(both)))
    
def info(train_data, val_data=None):
    print("Train: ")
    print_sample_count(train_data[0], train_data[1],
                       train_data[2], train_data[3])
    if val_data:
        print("\nTest: ")
        print_sample_count(val_data[0], val_data[1],
                           val_data[2], val_data[3])
    print("-----------------------")


def compute_recall(cm):
        precisions = []
        for i, row in enumerate(cm):
            if sum(row) == 0:
                precisions.append(0)
            else:
                precisions.append(row[i] / sum(row))
        return sum(precisions) / len(precisions)

def compute_precision(cm):
    recalls = []
    for i, row in enumerate(cm.T):
        if sum(row) == 0:
            recalls.append(0)
        else:
            recalls.append(row[i] / sum(row))
    return sum(recalls) / len(recalls)

def compute_class_accuracies(cm):
    accs = []
    for i, row in enumerate(cm):
        accs.append(row[i] / sum(row))
    return accs

def visualize_spectrogram(spect, sr, name):
    fig = plt.figure(figsize=(20, 10))
    display.specshow(
        spect,
        # x_axis="time",
        sr=sr,
        cmap="coolwarm",
    )
    # plt.xticks(np.arange(0, 1250, 125))
    # plt.yticks(np.arange(0, 128, 32))
    plt.colorbar()
    plt.show()
    plt.savefig(name)
    plt.close()

    
def convert_single(element):
    if element[0] == 0 and element[1] == 0:
        return 0
    elif element[0] == 1 and element[1] == 0:
        return 1
    elif element[0] == 0 and element[1] == 1:
        return 2
    elif element[0] == 1 and element[1] == 1:
        return 3

def label_data(validation_data):
    labels_sequence = []
    for element in validation_data:
        if element[1] == 0 and element[2] == 0:
            labels_sequence.append(0)
        elif element[1] == 1 and element[2] == 0:
            labels_sequence.append(1)
        elif element[1] == 0 and element[2] == 1:
            labels_sequence.append(2)
        elif element[1] == 1 and element[2] == 1:
            labels_sequence.append(3)
    return labels_sequence

def convert(inp):
    
    out = []
    for element in inp:
        if element[0] == 0 and element[1] == 0:
            out.append(0)
        elif element[0] == 1 and element[1] == 0:
            out.append(1)
        elif element[0] == 0 and element[1] == 1:
            out.append(2)
        elif element[0] == 1 and element[1] == 1:
            out.append(3)
    return out

def multi_hot_encoding(sample):
    # [crackles, wheezes]
    if sample == 1:
        return [1, 0]
    elif sample == 2:
        return [0, 1]
    elif sample == 3:
        return [1, 1]
    return [0, 0]

def oversample_dataset(data):

    original_length = sum([len(data[i]) for i in range(0, len(data))])
    print("Original length of training set: {}".format(original_length))
    majority_class = data[0]
    minority_class = data[1] + data[2] + data[3]
    print("Majority class: {} and Minority classes: {}".format(
        len(majority_class), original_length - len(majority_class)))

    ids = []
    for i in range(0, len(minority_class)):
        ids.append(i)
    choices = np.random.choice(
        ids, len(majority_class) - len(minority_class))
    
    resampled_minority = []
    for element in choices:
        resampled_minority.append(minority_class[element])
    print("Number of Elements to add: {}".format(
        len(resampled_minority) - len(minority_class)))

    resampled_minority += minority_class
    # resampled_minority += data[3]

    resampled_all = np.concatenate(
        [resampled_minority, majority_class], axis=0)
    print("Final length of training set: {}".format(len(resampled_all)))

    # shuffle et voila!

    order = np.arange(len(resampled_all))
    np.random.shuffle(order)
    resampled_all = resampled_all[order]
    data = resampled_all
    return data

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def load_text(spec):
    spec = np.loadtxt(spec, delimiter=',')
    return spec
    
def write_to_txt_file(spec, destination, file_name):

    file_path = os.path.join(destination, file_name)
    np.savetxt(file_path, spec, delimiter=',')
    return file_path
