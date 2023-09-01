import sys
from tensorflow.python.lib.io import file_io
import pickle
import numpy as np
import librosa
from librosa import display
from matplotlib import pyplot as plt

sys.path.append(
    '/Users/alirachidi/Documents/Sonavi_Labs/classification_algorithm/scripts/dataset_generation')
from generate_dataset import *

def visualize_spectrogram(spect, sr):
    fig = plt.figure(figsize=(20, 10))
    display.specshow(
        spect,
        # y_axis="log",
        sr=sr,
        cmap="coolwarm"
    )
    plt.colorbar()
    plt.show()
    
def generate_hist(datasets, sr):
    all_specs = []
    for dataset in datasets:
        all_specs += dataset
    print(len(all_specs))
    hist = np.zeros(all_specs[0][0].shape)
    for spec in all_specs:
        for i in range(0, spec[0].shape[0]):
            for y in range(0, spec[0].shape[1]):
                hist[i][y] += spec[0][i][y]
    
    hist = hist / len(all_specs)
    visualize_spectrogram(hist, sr)
    print("Done.")
    
def main():
    sr = 8000
    train_file = '/Users/alirachidi/Documents/Sonavi_Labs/classification_algorithm/data/datasets/all_sw_coch_preprocessed_v2_param_v4_augm_v0_cleaned_8000.pkl'
    file_stream = file_io.FileIO(train_file, mode="rb")
    data = pickle.load(file_stream)

    none_train, c_train, w_train, c_w_train = [
        data[0][i] for i in range(0, len(data[0]))
    ]
    none_test, c_test, w_test, c_w_test = [
        data[1][i] for i in range(0, len(data[1]))
    ]

    # data preparation
    train_data = [
        sample
        for label in [none_train, c_train, w_train, c_w_train]
        for sample in label
    ]
    validation_data = [
        sample for label in [none_test, c_test, w_test, c_w_test] for sample in label
    ]
    
    # generate_hist([none_train, none_test], sr)
    # generate_hist([c_train, c_test], sr)
    # generate_hist([w_train, w_test], sr)
    generate_hist([c_w_train, c_w_test], sr)
    
    # create spec
    
    
    
if __name__ == "__main__":
    main()