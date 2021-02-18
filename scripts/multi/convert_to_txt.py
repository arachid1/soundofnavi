import tensorflow as tf
import pickle
import librosa
from scipy.signal import lfilter
from librosa import display
import numpy as np
import os
import xlwt 
import multiprocessing
from xlwt import Workbook 
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from tensorflow.python.lib.io import file_io
from tensorflow.keras.layers import Layer, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.python.keras import backend
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras import activations
import tensorflow.keras.backend as K
from vis.visualization import visualize_saliency
from vis.utils import utils
from tf_explain.core.grad_cam import GradCAM
from sklearn.metrics import confusion_matrix

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

def generate_padded_samples(source, output_length):
    output_length = int(output_length)
    copy = np.zeros(output_length, dtype=np.float32)
    src_length = len(source)
    frac = src_length / output_length
    if(frac < 0.5):
        # tile forward sounds to fill empty space
        cursor = 0
        while(cursor + src_length) < output_length:
            copy[cursor:(cursor + src_length)] = source[:]
            cursor += src_length
    else:
        copy[:src_length] = source[:]
    #
    return copy

def convert_output(pred):
    if pred[0] > 0.5:
        if pred[1] > 0.5:
            return "both"
        else: 
            return "crackles"
    else:
        if pred[1] > 0.5:
            return "wheezes"
        else:
            return "normal"

def generate_cochlear_spec(x, coch_params, coch_b, coch_a, order):

    # get filter bank
    L, M = coch_b.shape
    # print("Shape: [{}, {}]".format(L, M))
    L_x = len(x)

    # octave shift, nonlinear factor, frame length, leaky integration
    shft = coch_params["shft"]  # octave shift (Matlab index: 4)
    fac = coch_params["fac"]  # nonlinear factor (Matlab index: 3)
    frmlen = coch_params["frmlen"]  # (Matlab index: 1)
    L_frm = np.round(frmlen * (2**(4+shft)))  # frame length (points)
    tc = coch_params["tc"]  # time constant (Matlab index: 2)
    tc = np.float64(tc)
    alph_exponent = -(1/(tc*(2**(4+shft))))
    alph = np.exp(alph_exponent)  # decaying factor

    # get data, allocate memory for output
    N = np.ceil(L_x/L_frm)  # number of frames
    # print("Number of frames: {}".format(N))
    # x = generate_padded_samples(x, ) # TODO: come back to padding

    v5 = np.zeros([int(N), M - 1])

    ##############################
    # last channel (highest frequency)
    ##############################

    # get filters from stored matrix
    # print("Number of filters: {}".format(M))
    p = int(order[M-1])  # ian's change 11/6

    # ian changed to 0 because of the way p, coch_a, and coch_b are seperated
    B = coch_b[0:p+1, M - 1]  # M-1 before
    A = coch_a[0:p+1, M - 1]
    y1 = lfilter(B, A, x)
    y2 = y1
    y2_h = y2

    # All other channels
    for ch in range(M-2, -1, -1):

        # ANALYSIS: cochlear filterbank
        # IIR: filter bank convolution ---> y1
        p = int(order[ch])
        B = coch_b[0:p+1, ch]
        A = coch_a[0:p+1, ch]

        y1 = lfilter(B, A, x)

        # TRANSDUCTION: hair cells
        y2 = y1

        # REDUCTION: lateral inhibitory network
        # masked by higher (frequency) spatial response
        y3 = y2 - y2_h
        y2_h = y2

        # half-wave rectifier
        y4 = y3.copy()
        y4[y4 < 0] = 0

        # temporal integration window #
        # leaky integration. alternative could be simpler short term average
        y5 = lfilter([1], [1-alph], y4)

        v5_row = []
        for i in range(1, int(N) + 1):
            v5_row.append(y5[int(L_frm*i) - 1])
        v5[:, ch] = v5_row
        # v5[:, ch] = y5[int(L_frm):int(L_frm*N)] # N elements, each space with L_frm
    # print("...End")

    return v5

def read_output_and_viz(model, inp, sr, i):

    sample = np.repeat(inp[0][..., np.newaxis], 3, -1)

    y = model.predict(np.array([sample]))

    print("i: {}".format(i))
    print(y)

    fig = plt.figure(figsize=(20, 10))
    display.specshow(
        inp[0],
        # y_axis="log",
        sr=sr,
        cmap="coolwarm"
    )
    plt.colorbar()
    plt.show()

    fig.savefig('{}.png'.format(i))

def send_to_txt(all_data, dataset_root):

    f = open(os.path.join(dataset_root, "paths_and_labels.txt"), 'w')
    for i, element in enumerate(all_data):
        file_path = os.path.join(dataset_root, "{}.txt".format(i))
        np.savetxt(file_path, element[0], delimiter=',')
        # arr = np.loadtxt(file_path, delimiter=',')
        # print(arr.shape)
        # exit()
        f.write('{}, {}, {}\n'.format(file_path, element[1], element[2]))
    f.close()

def main():
    
    file_name = 'perch_sw_coch_param_v14_augm_v0_8000'
    ## data retrieval
    train_file ='../../data/datasets/{}.pkl'.format(file_name)
    file_stream = file_io.FileIO(train_file, mode="rb")
    data = pickle.load(file_stream)

    dataset_root = '../../data/txt_datasets/{}'.format(file_name)

    info(data[0], data[1])

    all_data = [
            sample
            for cat in data
            for label in cat
            for sample in label
    ]
    if not os.path.exists(dataset_root):
        os.mkdir(dataset_root)

    #
    send_to_txt(all_data, dataset_root)


if __name__ == "__main__":
    main()