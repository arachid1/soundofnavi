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
    print("yo")


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

def visualize_spectrogram(spect, sr, name):
    fig = plt.figure(figsize=(20, 10))
    display.specshow(
        spect,
        # y_axis="log",
        sr=sr,
        cmap="coolwarm"
    )
    plt.colorbar()
    plt.show()
    plt.savefig(name)

    
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


def multi_hot_encoding(sample):
    # [crackles, wheezes]
    if sample == 1:
        return [1, 0]
    elif sample == 2:
        return [0, 1]
    elif sample == 3:
        return [1, 1]
    return [0, 0]

def parse_function(filename, label, shape):

    spectrogram = tf.io.read_file(filename)
    # arr2 = tf.strings.split(arr, sep=',')
    spectrogram = tf.strings.split(spectrogram)
    # arr3 = tf.strings.unicode_decode(arr3, 'UTF-8')
    # print(arr2[:128])
    # print(arr3[0])
    spectrogram = tf.strings.split(spectrogram, sep=',')
    # print(tf.size(arr3))
    # print(type(arr3))
    spectrogram =tf.strings.to_number(spectrogram)
    spectrogram = tf.reshape(spectrogram.to_tensor(), (shape[0], shape[1]))
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    spectrogram = tf.tile(spectrogram, [1, 1, 3])
    return spectrogram, label

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