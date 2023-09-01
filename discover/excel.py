import tensorflow as tf
import pickle
import librosa
from scipy.signal import lfilter
from librosa import display
import numpy as np
import os
import sys
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
from keras.models import load_model
from tensorflow.keras import activations
from sklearn.metrics import confusion_matrix

sys.path.append("/home/alirachidi/classification_algorithm/trainers/main/models/conv/")

from core import *

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

# class InvertedResidual(Layer):
#     def __init__(self, filters, strides, activation=ReLU(), kernel_size=3, expansion_factor=6,
#                  regularizer=None, trainable=True, name=None, **kwargs):
#         super(InvertedResidual, self).__init__(
#             trainable=trainable, name=name, **kwargs)
#         self.filters = filters
#         self.strides = strides
#         self.kernel_size = kernel_size
#         self.expansion_factor = expansion_factor
#         self.activation = activation
#         self.regularizer = regularizer
#         self.channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

#     def build(self, input_shape):
#         input_channels = int(input_shape[self.channel_axis])  # C
#         self.ptwise_conv1 = Conv2D(filters=int(input_channels*self.expansion_factor),
#                                    kernel_size=1, kernel_regularizer=self.regularizer, use_bias=False)
#         self.dwise = DepthwiseConv2D(kernel_size=self.kernel_size, strides=self.strides,
#                                      kernel_regularizer=self.regularizer, padding='same', use_bias=False)
#         self.ptwise_conv2 = Conv2D(filters=self.filters, kernel_size=1,
#                                    kernel_regularizer=self.regularizer, use_bias=False)
#         self.bn1 = BatchNormalization()
#         self.bn2 = BatchNormalization()
#         self.bn3 = BatchNormalization()
        
#         # input = C 
#         # 1x1 -> 4C
#         # depthwise 3x3 -> 4C
#         # 1x1 -> C 
#         # ouput = C

#     def call(self, input_x, training=False):
#         # Expansion
#         x = self.ptwise_conv1(input_x)
#         x = self.bn1(x, training=training)
#         x = self.activation(x)
#         # Spatial filtering
#         x = self.dwise(x)
#         x = self.bn2(x, training=training)
#         x = self.activation(x)
#         # back to low-channels w/o activation
#         x = self.ptwise_conv2(x)
#         x = self.bn3(x, training=training)
#         # Residual connection only if i/o have same spatial and depth dims
#         if input_x.shape[1:] == x.shape[1:]:
#             x += input_x
#         return x

#     def get_config(self):
#         cfg = super(InvertedResidual, self).get_config()
#         cfg.update({'filters': self.filters,
#                     'strides': self.strides,
#                     'regularizer': self.regularizer,
#                     'expansion_factor': self.expansion_factor,
#                     'activation': self.activation})
#         return cfg


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


def return_mod9(SHAPE, BATCH_SIZE, N_CLASSES):

    KERNEL_SIZE = 6
    POOL_SIZE = (2, 2)
    LL2_REG = 0
    i = layers.Input(shape=SHAPE, batch_size=BATCH_SIZE)
    x = layers.BatchNormalization()(i)
    tower_1 = layers.Conv2D(16, (1,1), padding='same', activation='relu')(x)
    tower_1 = layers.Conv2D(16, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = layers.Conv2D(16, (1,1), padding='same', activation='relu')(x)
    tower_2 = layers.Conv2D(16, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    tower_3 = layers.Conv2D(16, (1,1), padding='same', activation='relu')(tower_3)
    x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = InvertedResidual(filters=32, strides=1, kernel_size=KERNEL_SIZE)(x)
    x = InvertedResidual(filters=32, strides=1, kernel_size=KERNEL_SIZE)(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = InvertedResidual(filters=64, strides=1, kernel_size=KERNEL_SIZE,)(x)
    x = InvertedResidual(filters=64, strides=1, kernel_size=KERNEL_SIZE,)(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = InvertedResidual(filters=128, strides=1, kernel_size=KERNEL_SIZE,)(x)
    x = InvertedResidual(filters=128, strides=1, kernel_size=KERNEL_SIZE,)(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = InvertedResidual(filters=256, strides=1, kernel_size=KERNEL_SIZE)(x)
    x = InvertedResidual(filters=256, strides=1, kernel_size=KERNEL_SIZE)(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = InvertedResidual(filters=512, strides=1, kernel_size=KERNEL_SIZE)(x)
    x = InvertedResidual(filters=512, strides=1, kernel_size=KERNEL_SIZE)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = DenseMoE(512, 2, expert_activation='relu', gating_activation='sigmoid')(x)
    o = layers.Dense(N_CLASSES, activity_regularizer=l2(
        LL2_REG), activation="sigmoid")(x)
    model = Model(inputs=i, outputs=o, name="conv2d")
    return model

def return_mod10(SHAPE, BATCH_SIZE, N_CLASSES):
    KERNEL_SIZE = (3, 3)
    POOL_SIZE = (2, 2)
    PADDING = "same"
    CHANNELS = 32
    DROPOUT = 0.1
    DENSE_LAYER = 32
    LL2_REG = 0
    i = layers.Input(shape=SHAPE)
    # dummy_layer = layers.Lambda(lambda x: x)(i)
    x = layers.BatchNormalization()(i)
    tower_1 = layers.Conv2D(8, (1,1), padding='same', activation='relu')(x)
    tower_1 = layers.Conv2D(8, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = layers.Conv2D(8, (1,1), padding='same', activation='relu')(x)
    tower_2 = layers.Conv2D(8, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    tower_3 = layers.Conv2D(8, (1,1), padding='same', activation='relu')(tower_3)
    x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    tower_1 = layers.Conv2D(16, (1,1), padding='same', activation='relu')(x)
    tower_1 = layers.Conv2D(16, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = layers.Conv2D(16, (1,1), padding='same', activation='relu')(x)
    tower_2 = layers.Conv2D(16, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    tower_3 = layers.Conv2D(16, (1,1), padding='same', activation='relu')(tower_3)
    x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    tower_1 = layers.Conv2D(32, (1,1), padding='same', activation='relu')(x)
    tower_1 = layers.Conv2D(32, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = layers.Conv2D(32, (1,1), padding='same', activation='relu')(x)
    tower_2 = layers.Conv2D(32, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    tower_3 = layers.Conv2D(32, (1,1), padding='same', activation='relu')(tower_3)
    x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    tower_1 = layers.Conv2D(64, (1,1), padding='same', activation='relu')(x)
    tower_1 = layers.Conv2D(64, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = layers.Conv2D(64, (1,1), padding='same', activation='relu')(x)
    tower_2 = layers.Conv2D(64, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    tower_3 = layers.Conv2D(64, (1,1), padding='same', activation='relu')(tower_3)
    x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    tower_1 = layers.Conv2D(128, (1,1), padding='same', activation='relu')(x)
    tower_1 = layers.Conv2D(128, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = layers.Conv2D(128, (1,1), padding='same', activation='relu')(x)
    tower_2 = layers.Conv2D(128, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    tower_3 = layers.Conv2D(128, (1,1), padding='same', activation='relu')(tower_3)
    x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    tower_1 = layers.Conv2D(256, (1,1), padding='same', activation='relu')(x)
    tower_1 = layers.Conv2D(256, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = layers.Conv2D(256, (1,1), padding='same', activation='relu')(x)
    tower_2 = layers.Conv2D(256, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    tower_3 = layers.Conv2D(256, (1,1), padding='same', activation='relu')(tower_3)
    x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    o = layers.Dense(N_CLASSES, activity_regularizer=l2(
        LL2_REG), activation="sigmoid")(x)
    model = Model(inputs=i, outputs=o, name="conv2d")
    return model

def return_mixnet(SHAPE, BATCH_SIZE, N_CLASSES):
    KERNELS = [3, 4, 5, 6, 7]
    POOL_SIZE=2

    i = layers.Input(shape=SHAPE, )
    x = MixDepthGroupConvolution2D(kernels=KERNELS)(i)
    x = layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = MixDepthGroupConvolution2D(kernels=KERNELS)(x)
    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = MixDepthGroupConvolution2D(kernels=KERNELS)(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = MixDepthGroupConvolution2D(kernels=KERNELS)(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = MixDepthGroupConvolution2D(kernels=KERNELS)(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    o = layers.Dense(N_CLASSES, activation="sigmoid")(x)

    model = Model(inputs=i, outputs=o, name="conv2d")
    return model


def generate_excel(root, model, excel_dest, coch_path):
    coch_a = np.loadtxt(os.path.join(coch_path,'COCH_A.txt'), delimiter=',')
    coch_b = np.loadtxt(os.path.join(coch_path,'COCH_B.txt'), delimiter=',')
    order = np.loadtxt(os.path.join(coch_path,'p.txt'), delimiter=',')
    fs = 8000
    bp = 1
    coch_params = {
        "frmlen": 8*(1/(fs/8000)), 
        "tc": 8, 
        "fac": -2, 
        "shft": np.log2(fs/16000), 
        "FULLT": 0, 
        "FULLX": 0, 
        "bp": bp
    }
    patient_ids = []

    count = 1

    print("Starting...")
    wb = Workbook() 
    sheet1 = wb.add_sheet('Sheet 1')
    for patient_id in os.listdir(root):
        patient_root = os.path.join(root, patient_id)
        raw_audios = []
        for file_ in os.listdir(patient_root):
            if not file_.endswith('.wav'):
                continue
            name = file_.split('/')[-1]
            raw_audios.append(name)
        raw_audios = sorted(raw_audios)
        sheet1.write(count, 0, patient_id)
        count_2 = 0
        for raw_audio in raw_audios:
            data, rate = librosa.load(os.path.join(patient_root, raw_audio), 8000)
            if not (len(data) == 80000):
                print("Length of {} is {}".format(os.path.join(patient_root, raw_audio), len(data)))
                data = generate_padded_samples(data, 80000)
            data = generate_cochlear_spec(data, coch_params, coch_b, coch_a, order)
            data = np.transpose(data)
            data = np.repeat(data[..., np.newaxis], 3, -1)
            output = model.predict(np.array([data,]))
            prediction = convert_output(output[0])
            sheet1.write(count, count_2 + 1, raw_audio) 
            sheet1.write(count, count_2 + 2, prediction) 
            sheet1.write(count, count_2 + 3, str(output[0][0])) 
            sheet1.write(count, count_2 + 4, str(output[0][1]))
            count_2 += 4
        count += 1
    print("Ended. Saving the Excel Sheet {}...".format(excel_dest))
    wb.save(excel_dest) 
    print("Done.")

def analyze_excel(filepath):
    # filepath = "../data/excel_spreadsheets/2026.xls"
    print("File: {}".format(filepath))
    s = pd.read_excel(filepath)
    print(s.stack().value_counts()[:4])
    print("Nb of elements: {}".format(s.stack().value_counts()[:4].sum()))

def main():

    # model retrieval
    coch_path = '../cochlear_preprocessing/128_channels/'
    # root = '../data/raw_audios/Bangladesh_wo_details/'
    root = '../data/raw_audios/Bangladesh_sounds_speech_removed_2/'
    filepath = '../cache/conv___03_0848___all_sw_coch_preprocessed_v2_param_v26_augm_v0_cleaned_8000___mixed_net___2014/model.h5'
    # filepath = '../cache/conv___03_1000___all_sw_coch_preprocessed_v2_param_v26_augm_v0_cleaned_8000___redo_w_newdataset___2011/model.h5'
    
    excel_dest = '../data/excel_spreadsheets/2014.xls'

    print("Root: {}".format(root))
    print("Filepath: {}".format(filepath))
    print("Excel_dest: {}".format(excel_dest))
    print("coch_path: {}".format(coch_path))

    sr = 8000
    SHAPE = (128, 1250, 3)
    BATCH_SIZE = 1
    N_CLASSES = 2

    # model = return_mod9(SHAPE, BATCH_SIZE, N_CLASSES)
    # model = return_mod10(SHAPE, BATCH_SIZE, N_CLASSES)
    model = return_mixnet(SHAPE, BATCH_SIZE, N_CLASSES)
    
    model.load_weights(filepath)
    # model = load_model(filepath)
    model.summary()

    ##### Excet sheet generation
    generate_excel(root, model, excel_dest, coch_path)

    analyze_excel(excel_dest)

if __name__ == "__main__":
    main()