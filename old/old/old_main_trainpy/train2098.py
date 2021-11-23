from .modules.helpers import *
from .modules.generators import *
from .modules.metrics import *
from .modules.callbacks import *
from .core import *

import argparse
import pickle
import tensorflow as tf
import sys
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
from sklearn.utils import class_weight
from tensorflow.python.lib.io import file_io
import json
import time
import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt
import librosa
from librosa import display
from PIL import Image
import xlwt
from io import BytesIO
from xlwt import Workbook 

def convert_pneumonia_label(label):
    if label == "NO PEP":
        return 0
    elif label == "PEP":
        return 1

def inverse_convert_pneumonia_label(label):
    if label == 0:
        return "NO PEP"
    elif label == 1:
        return "PEP"

def parse_function_for_val(filename, label, shape):

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
    # tf.print(label)
    return spectrogram, label, filename

def conv2d(N_CLASSES, SR, BATCH_SIZE, LR, SHAPE, WEIGHT_DECAY, LL2_REG, EPSILON, LABEL_SMOOTHING):

    # KERNELS = [3, 4, 5, 6, 7]
    # POOL_SIZE=2
    # i = layers.Input(shape=SHAPE, )
    # x = layers.Conv2D(8, (3, 3), padding='same', activation='relu')(i)
    # x = MixDepthGroupConvolution2D(kernels=KERNELS)(x)
    # x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    # x = MixDepthGroupConvolution2D(kernels=KERNELS)(x)
    # x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    # x = MixDepthGroupConvolution2D(kernels=KERNELS)(x)
    # x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    # x = MixDepthGroupConvolution2D(kernels=KERNELS)(x)
    # x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    # x = MixDepthGroupConvolution2D(kernels=KERNELS)(x)
    # x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.GlobalAveragePooling2D()(x)
    # o = layers.Dense(N_CLASSES, activity_regularizer=l2(
    #     LL2_REG), activation="sigmoid")(x)

    # KERNEL_SIZE = (3, 3)
    # POOL_SIZE = (2, 2)
    # PADDING = "same"
    # CHANNELS = 32
    # DROPOUT = 0.1
    # DENSE_LAYER = 32
    # i = layers.Input(shape=SHAPE,)
    # x = layers.BatchNormalization()(i)
    # tower_1 = layers.Conv2D(8, (1,1), padding='same', activation='relu')(x)
    # tower_1 = layers.Conv2D(8, (3,3), padding='same', activation='relu')(tower_1)
    # tower_2 = layers.Conv2D(8, (1,1), padding='same', activation='relu')(x)
    # tower_2 = layers.Conv2D(8, (5,5), padding='same', activation='relu')(tower_2)
    # tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    # tower_3 = layers.Conv2D(8, (1,1), padding='same', activation='relu')(tower_3)
    # x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    # x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    # x = layers.BatchNormalization()(x)
    # tower_1 = layers.Conv2D(16, (1,1), padding='same', activation='relu')(x)
    # tower_1 = layers.Conv2D(16, (3,3), padding='same', activation='relu')(tower_1)
    # tower_2 = layers.Conv2D(16, (1,1), padding='same', activation='relu')(x)
    # tower_2 = layers.Conv2D(16, (5,5), padding='same', activation='relu')(tower_2)
    # tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    # tower_3 = layers.Conv2D(16, (1,1), padding='same', activation='relu')(tower_3)
    # x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    # x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    # x = layers.BatchNormalization()(x)
    # tower_1 = layers.Conv2D(32, (1,1), padding='same', activation='relu')(x)
    # tower_1 = layers.Conv2D(32, (3,3), padding='same', activation='relu')(tower_1)
    # tower_2 = layers.Conv2D(32, (1,1), padding='same', activation='relu')(x)
    # tower_2 = layers.Conv2D(32, (5,5), padding='same', activation='relu')(tower_2)
    # tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    # tower_3 = layers.Conv2D(32, (1,1), padding='same', activation='relu')(tower_3)
    # x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    # x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    # x = layers.BatchNormalization()(x)
    # tower_1 = layers.Conv2D(64, (1,1), padding='same', activation='relu')(x)
    # tower_1 = layers.Conv2D(64, (3,3), padding='same', activation='relu')(tower_1)
    # tower_2 = layers.Conv2D(64, (1,1), padding='same', activation='relu')(x)
    # tower_2 = layers.Conv2D(64, (5,5), padding='same', activation='relu')(tower_2)
    # tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    # tower_3 = layers.Conv2D(64, (1,1), padding='same', activation='relu')(tower_3)
    # x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    # x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    # x = layers.BatchNormalization()(x)
    # tower_1 = layers.Conv2D(128, (1,1), padding='same', activation='relu')(x)
    # tower_1 = layers.Conv2D(128, (3,3), padding='same', activation='relu')(tower_1)
    # tower_2 = layers.Conv2D(128, (1,1), padding='same', activation='relu')(x)
    # tower_2 = layers.Conv2D(128, (5,5), padding='same', activation='relu')(tower_2)
    # tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    # tower_3 = layers.Conv2D(128, (1,1), padding='same', activation='relu')(tower_3)
    # x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    # x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    # x = layers.BatchNormalization()(x)
    # tower_1 = layers.Conv2D(256, (1,1), padding='same', activation='relu')(x)
    # tower_1 = layers.Conv2D(256, (3,3), padding='same', activation='relu')(tower_1)
    # tower_2 = layers.Conv2D(256, (1,1), padding='same', activation='relu')(x)
    # tower_2 = layers.Conv2D(256, (5,5), padding='same', activation='relu')(tower_2)
    # tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    # tower_3 = layers.Conv2D(256, (1,1), padding='same', activation='relu')(tower_3)
    # x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    # x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.GlobalAveragePooling2D()(x)
    # o = layers.Dense(N_CLASSES, activity_regularizer=l2(
    #     LL2_REG), activation="sigmoid")(x)

    KERNEL_SIZE = 6
    POOL_SIZE = (2, 2)
    i = layers.Input(shape=SHAPE, )
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
    o = layers.Dense(N_CLASSES, activity_regularizer=l2(
        LL2_REG), activation="sigmoid")(x)
    # delete above

    model = Model(inputs=i, outputs=o, name="conv2d")
    opt = tf.keras.optimizers.Adam(
        lr=LR, beta_1=0.9, beta_2=0.999, epsilon=EPSILON, decay=WEIGHT_DECAY, amsgrad=False
    )
    # class_acc = class_accuracy()
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING)
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=[
            'accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), gen_confusion_matrix
        ],
    )
    return model
    

def save_spectrogram(spec, name, sr):
    fig = plt.figure(figsize=(20, 10))
    display.specshow(
        spec,
        # y_axis="log",
        sr=sr,
        cmap="coolwarm"
    )
    plt.colorbar()
    plt.show()
    plt.savefig(name)

def train_model(train_file, **args):
    logs_path = job_dir + "/logs/"
    print("-----------------------")
    print("Using module located at {}".format(__file__[-30:]))
    print("Using train_file located at {}".format(train_file))
    print("Using logs_path located at {}".format(logs_path))
    print("-----------------------")

    # setting variables
    print("Collecting Variables...")

    n_classes = params["N_CLASSES"]

    n_epochs = params["N_EPOCHS"]

    sr = params["SR"]
    lr = params["LR"]
    batch_size = params["BATCH_SIZE"]

    ll2_reg = params["LL2_REG"]
    weight_decay = params["WEIGHT_DECAY"]
    label_smoothing = params["LABEL_SMOOTHING"]

    epsilon = params["EPSILON"]

    shape = tuple(params["SHAPE"])

    es_patience = params["ES_PATIENCE"]
    min_delta = params["MIN_DELTA"]

    initial_channels = params["INITIAL_CHANNELS"]
    shape = shape + (initial_channels, )
    
    model_params = {
        "N_CLASSES": n_classes,
        "SR": sr,
        "BATCH_SIZE": batch_size,
        "LR": lr,
        "SHAPE": shape,
        "WEIGHT_DECAY": weight_decay,
        "LL2_REG": ll2_reg,
        "EPSILON": epsilon,
        "LABEL_SMOOTHING": label_smoothing
    }

    factor = params["FACTOR"]
    patience = params["PATIENCE"]
    min_lr = params["MIN_LR"]

    lr_params = {
        "factor": factor,
        "patience": patience,
        "min_lr": min_lr
    }

    print("Model Parameters: {}".format(model_params))
    print("Learning Rate Parameters: {}".format(lr_params))
    print("Early Stopping Patience and Delta: {}, {}%".format(es_patience, min_delta*100))
    print("-----------------------")

    train_test_ratio = 0.8

    filenames = []
    labels = []

    # with tf.device('/CPU:0'): 

    #     train_file_name = train_file.split('/')[-1].split('.')[0]

    #     previous_filename = None

    #     root = "../../data/PCV_SEGMENTED_Processed_Files/"
    #     path = "/home/alirachidi/classification_algorithm/data/txt_datasets/all_sw_coch_preprocessed_v2_param_v29_augm_v0_cleaned_8000/"
    #     excel_path = "/home/alirachidi/classification_algorithm/data/Bangladesh_PCV_onlyStudyPatients.xlsx"
    #     file_number = "2087"
    #     excel_dest = "/home/alirachidi/classification_algorithm/data/excel_spreadsheets/Bangladesh_pneumonia_patient_decision_{}.xls".format(file_number)
    #     # model_path = "/home/alirachidi/classification_algorithm/cache/conv___04_1744___all_sw_coch_preprocessed_v2_param_v29_augm_v0_cleaned_8000___mixednet_sorted___2075/model.h5"
    #     # model_path = "/home/alirachidi/classification_algorithm/cache/conv___04_0121___all_sw_coch_preprocessed_v2_param_v29_augm_v0_cleaned_8000___test___2049/third.h5"   
    #     # model_path = "/home/alirachidi/classification_algorithm/cache/conv___04_1519___all_sw_coch_preprocessed_v2_param_v29_augm_v0_cleaned_8000___mod9_sorted_postshuffle___2078/model.h5"
    #     model_path = "/home/alirachidi/classification_algorithm/cache/conv___04_1413___all_sw_coch_preprocessed_v2_param_v29_augm_v0_cleaned_8000___resting_shuffled___{}/model.h5".format(file_number)

    #     print("excel dest: {}".format(excel_dest))
    #     print("model: {}".format(model_path))
    #     patient_recordings = []

    #     filenames = []
    #     labels = []
    #     df = pd.read_excel(excel_path, engine='openpyxl')

    #     print("Original number of folders: {}".format(len(glob.glob(os.path.join(root, "*")))))
    #     folder_paths = glob.glob(os.path.join(root, "*"),recursive=True)
    #     folder_paths = sorted(folder_paths)

    #     count_1 = 0
    #     count_2 = 0
    #     count_3 = 0

    #     for folder_path in folder_paths:
    #         if folder_path == os.path.join(root, "0365993_SEGMENTED") or folder_path == os.path.join(root, "0273320_SEGMENTED") or folder_path == os.path.join(root, "0364772_SEGMENTED"):
    #             print("here")
    #             continue
    #         recordings = glob.glob(os.path.join(folder_path, "*.wav"))
    #         recordings = sorted(recordings)
    #         # print(recordings)
    #         # exit()
    #         if len(recordings) != 6:
    #             # print("recordings not equal 6")
    #             count_1 += 1
    #             continue
    #         patient_id = int(folder_path.split('/')[-1].split('_')[0])
    #         file_column = df.loc[df['HOSP_ID'] == patient_id]
    #         if file_column.empty:
    #             # print("empty col")
    #             count_2 += 1
    #             continue
    #         label = str(file_column['PEP1'].values[0])
    #         if label == "Uninterpretable":
    #             count_3 += 1
    #             continue
    #         final_chunks = []
    #         for recording in recordings:
    #             recording_name = recording.split('/')[-1].split('.')[0]
    #             chunks = sorted(glob.glob(os.path.join(path, "{}*.txt".format(recording_name))))
    #             if len(chunks) > 1:
    #                 final_chunks.append(chunks[int(len(chunks)/2)])
    #             else:
    #                 final_chunks.append(chunks[0])
    #         filenames.append(final_chunks)
    #         labels.append(convert_pneumonia_label(label))

    #     # print(count_1)
    #     # print(count_2)
    #     # print(count_3)

    #     print("Expected number of patients: {}".format(len(filenames)))

    #     # weights
    #     weights = None

    #     if bool(params["CLASS_WEIGHTS"]):
    #         print("Initializing weights...")
    #         # y_train = []
    #         # for label in labels:
    #         #     y_train.append(convert_single(label))
    #         weights = class_weight.compute_class_weight(
    #             "balanced", np.unique(labels), labels)

    #         weights = {i: weights[i] for i in range(0, len(weights))}
    #         print("weights = {}".format(weights))
        
    #     # callbacks
    #     # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=1)
    #     tensorboard_callback = lr_tensorboard(log_dir=logs_path, histogram_freq=1)
    #     reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    #         monitor="val_accuracy", verbose=1, **lr_params
    #     )
    #     early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    #         monitor='val_accuracy', min_delta=min_delta, patience=es_patience, verbose=1,
    #     )

    # gpus = tf.config.experimental.list_logical_devices('GPU')

    # # model setting
    # model = conv2d(**model_params)

    # model.load_weights(model_path)

    # model.summary()

    print("Starting...")
    wb = Workbook() 
    sheet1 = wb.add_sheet('Sheet_1', )

    col_width = 256*20
    col_height = 256*40

    sheet1.row(0).height_mismatch = True
    sheet1.col(0).width_mismatch = True

    sheet1.col(0).width = col_width
    sheet1.row(0).height = col_height

    img = Image.open('/home/alirachidi/classification_algorithm/trainers/main/train_2077.png').convert('RGB')
    fo = BytesIO()
    img.save(fo, format='bmp')
    # print(img.size)
    sheet1.insert_bitmap_data(fo.getvalue(),0,0, scale_x = 0.05, scale_y = 0.05)
    wb.save('test.xls')
    img.close()

    exit()

    row = 1
    col = 0
    true_positives = 0
    sheet1.write(0, 0, "Patient ID") 
    sheet1.write(0, 1, "File name") 
    sheet1.write(0, 2, "Body area level decision") 
    sheet1.write(0, 3, "Body area level decision (count)") 
    sheet1.write(0, 4, "Patient level decision") 
    sheet1.write(0, 5, "Label") 

    # print(filenames[:5])
    # print(filenames[::-1][:5])
    # exit()
    
    filenames = filenames[::-1]
    val_index = int(len(filenames)/5)
    avg_spectrogram = np.zeros((shape[0], shape[1]))

    for i, filename in enumerate(filenames):
            name = filename[0].split('/')[-1][:9]
            counter = 0
            file_column = df.loc[df['HOSP_ID'] == patient_id]
            for extract in filename:
                sheet1.write(row, 0, name) 
                sheet1.write(row, 1, extract.split('/')[-1]) 
                arr = np.loadtxt(extract, delimiter=',')
                avg_spectrogram += arr
                if i > val_index:
                    set_avg_spectrogram += arr
                arr = np.repeat(arr[..., np.newaxis], 3, -1)
                output = model.predict(np.array([arr]))
                output = round(output[0][0])
                if output == 1:
                    counter += 1
                sheet1.write(row, 2, inverse_convert_pneumonia_label(output))
                row += 1
            # calculating accuracy 
            if ((counter > 0) and (labels[i] == 1)) or ((counter == 0) and (labels[i] == 0)):
                true_positives += 1 
            sheet1.write(row, 3, counter)
            # setting prediction
            if counter > 0:
                sheet1.write(row, 4, "PEP")
            else:
                sheet1.write(row, 4, "NO PEP")
            sheet1.write(row, 5, inverse_convert_pneumonia_label(labels[i]))
            row += 1
            if i == val_index:
                set_avg_spectrogram = avg_spectrogram / (i+1)
                save_spectrogram(set_avg_spectrogram, "val_avg_{}".format(file_number), sr)
                print("accuracy at {} elements: {}".format(i+1, true_positives/(i+1)))
                set_avg_spectrogram = np.zeros((shape[0], shape[1]))

    avg_spectrogram = avg_spectrogram / len(filenames)
    save_spectrogram(avg_spectrogram, "all_avg_{}".format(file_number), sr)

    print("Accuracy (of {} elements): {}".format(len(filenames), true_positives/len(filenames)))
    print("Ended. Saving the Excel Sheet {}...".format(excel_dest))
    wb.save(excel_dest) 
    print("Done.")


    # for spectrogram, label, filename in val_dataset:
        # current_filename = filename.numpy()[0].decode('ascii').split('/')[-1]
        # if previous_filename == None:
        #     patient.append((spectrogram, label)
        #     continue



        # # if current_filename[:8] == previous_filename or previous_filename == None: # same: gotta continue processing
        # #     continue
        # else: 
        #     patient = []
        #     process_all()
        # previous_filename = current_filename[:8]
        # if int(current_filename[-9]) == (len(patient) + 1): # new number so add
        #     patient.append((spectrogram, label))
        # else:
        #     continue
        # exit()

    # if gpus:
    #     for gpu in gpus:
    #         model.fit(
    #             train_dataset,
    #             validation_data=val_dataset,
    #             epochs=n_epochs,
    #             verbose=2,
    #             class_weight=weights,
    #             callbacks=[tensorboard_callback, reduce_lr_callback, early_stopping_callback]
    #         )
    


    # model.save(job_dir + "/model.h5")

if __name__ == "__main__":
    print("Tensorflow Version: {}".format(tf.__version__))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    # tf.debugging.set_log_device_placement(True)
    # tf.debugging.set_log_device_placement(True)
    seed_everything()
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
        "--train-file",
        help="GCS or local paths to training data",
        required=True
    )

    parser.add_argument(
        "--job-dir",
        help="GCS location to write checkpoints and export models",
        required=True,
    )
    parser.add_argument(
        "--params",
        help="parameters used in the model and training",
        required=True,
    )
    args = parser.parse_args()
    arguments = args.__dict__
    job_dir = arguments.pop("job_dir")
    params = arguments.pop("params")
    params = json.loads(params)
    train_model(**arguments)
