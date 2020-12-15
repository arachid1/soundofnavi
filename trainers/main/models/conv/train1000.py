import argparse
import os
import random
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras import metrics
from tensorflow.keras.regularizers import l2

from tensorflow.python.keras import backend as K
from tensorflow.python.lib.io import file_io
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight

import keras.utils.generic_utils

import pickle
from matplotlib import pyplot as plt
import itertools
import librosa
from librosa import display
import soundfile as sf
import io

import nlpaug.flow as naf
import nlpaug.augmenter.spectrogram as nas

import json
from .core import *
# import sys

# sys.path.insert(
#     1,
#     "/Users/alirachidi/Documents/Sonavi_Labs/ObjectDetection/trainers/normal/trainer_mfcc/modules/",
# )

# from my_metrics import class_accuracy, class_precision

print("Tensorflow Version: {}".format(tf.__version__))

K.clear_session()

################


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

#################
#################


def conv2d(N_CLASSES, SR, BATCH_SIZE, LR, SHAPE, WEIGHT_DECAY, LL2_REG, EPSILON):

    KERNEL_SIZE = (3, 3)
    POOL_SIZE = (2, 2)
    PADDING = "same"
    CHANNELS = 32
    DROPOUT = 0.1
    DENSE_LAYER = 32
    i = layers.Input(shape=SHAPE, batch_size=BATCH_SIZE)

    o = layers.Dense(N_CLASSES, activity_regularizer=l2(
        LL2_REG), activation="sigmoid")(x)
    # delete above

    model = Model(inputs=i, outputs=o, name="conv2d")
    opt = tf.keras.optimizers.Adam(
        lr=LR, beta_1=0.9, beta_2=0.999, epsilon=EPSILON, decay=WEIGHT_DECAY, amsgrad=False
    )
    class_acc = class_accuracy()
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[
            'accuracy'
        ],
    )
    return model


#################
#################


class class_accuracy(tf.keras.metrics.Metric):
    def __init__(self, name="accuracy", **kwargs):
        super(class_accuracy, self).__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name="acc", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(tf.round(y_pred), tf.int32)
        matches = tf.equal(y_true, y_pred)
        matches = tf.map_fn(
            lambda x: tf.equal(tf.reduce_sum(
                tf.cast(x, tf.float32)), 2), matches
        )
        matches = tf.cast(matches, tf.float32)
        acc = tf.reduce_sum(matches) / \
            tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
        self.accuracy.assign(acc)

    def result(self):
        return self.accuracy


#################
#################

# def visualise_inputs(model, dimension, batch_size):
#     inp = model.input                                           # input placeholder
#     outputs = [layer.output for layer in model.layers]          # all layer outputs
#     functor = K.function([inp, K.symbolic_learning_phase()], outputs )   # evaluation function

#     # Testing
#     test = np.random.random((batch_size, 50, dimension))[np.newaxis,...]
#     print(test.shape)
#     layer_outs = functor([test, 1.])
#     print(layer_outs)


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

#################
#################


class data_generator(tf.keras.utils.Sequence):
    def __init__(
        self, wav_files, sr, n_classes, shape, batch_size, initial_channels, shuffle=True,
    ):
        self.wav_files = wav_files
        # self.labels = self.all_labels(self.wav_files)
        self.n_classes = n_classes
        self.shape = shape
        self.batch_size = batch_size
        self.initial_channels = initial_channels
        self.timesteps = 0
        self.shuffle = True
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.wav_files) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index *
                               self.batch_size: (index + 1) * self.batch_size]
        wav_batch = [self.wav_files[k] for k in indexes]

        # returns labels and the longest timestamp
        wav_labels = label_data(wav_batch)

        final_shape = (self.batch_size,
                       self.shape[0], self.shape[1], self.initial_channels)

        X = np.zeros(final_shape, dtype=np.float32)
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (wav, label) in enumerate(zip(wav_batch, wav_labels)):
            wav = wav[0]
            wav = np.repeat(wav[..., np.newaxis], self.initial_channels, -1)
            X[i, :, :] = wav
            Y[i, ] = multi_hot_encoding(label)
        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

#################
#################
#################


class cm_callback(tf.keras.callbacks.LambdaCallback):
    def __init__(
        self, validation_data, shape, n_classes, sr, save, add_tuned, es_patience, min_delta
    ):
        self.shape = shape
        self.n_classes = n_classes
        self.validation_data = validation_data
        self.sr = sr
        self.save = save
        self.add_tuned = add_tuned
        self.es_patience = es_patience
        self.min_delta = min_delta
        self.num_of_val_samples = len(self.validation_data)
        self.X = self.prepare_data(self.validation_data)
        self.Y = label_data(self.validation_data)
        self.highest_acc = 0
        self.highest_acc_preds = None
        self.tracker_accuracy = 0
        self.tracker = 0
        self.confusion_matrix = None
        self.tuned_confusion_matrix = None
        self.training_accs = []
        self.samples_to_be_saved = []
        self.train_writer = tf.summary.create_file_writer(
            job_dir + "/logs/train_acc")
        self.validation_writer = tf.summary.create_file_writer(
            job_dir + "/logs/val_acc")
        self.none_writer = tf.summary.create_file_writer(
            job_dir + "/logs/none")
        self.crackles_writer = tf.summary.create_file_writer(
            job_dir + "/logs/crackles")
        self.wheezes_writer = tf.summary.create_file_writer(
            job_dir + "/logs/wheezes")
        self.both_writer = tf.summary.create_file_writer(
            job_dir + "/logs/both")
        self.class_names = ["none", "crackles", "wheezes", "both"]

    def on_batch_end(self, batch, logs=None):
        acc = logs['accuracy']
        self.training_accs.append(acc)
        
    def on_train_end(self, logs=None):

        print("Sending Confusion Matrices...")
        figure = self.plot_confusion_matrix(
            self.confusion_matrix, class_names=self.class_names
        )
        cm_image = self.plot_to_image(figure)

        if self.add_tuned:
            tuned_figure = self.plot_confusion_matrix(
                self.tuned_confusion_matrix, class_names=self.class_names
            )
            tuned_cm_image = self.plot_to_image(tuned_figure)

        with self.validation_writer.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=1)
            tf.summary.scalar("Highest accuracy", self.highest_acc, step=0)
            if self.add_tuned:
                tf.summary.image(
                    "Tuned Confusion Matrix", tuned_cm_image, step=1,
                )
        if self.save:
            print("...and saving audio and images...")
            self.collect_samples()
            self.save_png_and_wav()
        print("Model summary: {}".format(self.model.summary()))
        print("The highest accuracy: {}".format(self.highest_acc))
        print("End of Training.")

    def on_epoch_end(self, epoch, logs=None):
        """

        Args:
        epoch: default arg
        logs: default arg
        """
        Y_pred = self.model.predict(self.X)

        # Not tuned

        preds = np.zeros(Y_pred.shape)

        preds[Y_pred >= 0.5] = 1
        preds[Y_pred < 0.5] = 0

        preds = self.convert(preds)

        training_acc = sum(self.training_accs) / len(self.training_accs)
        print("Training accuracy: {}".format(training_acc))
        self.training_accs = []

        cm = confusion_matrix(self.Y, preds)
        print("cm: {}".format(cm))

        acc = np.trace(cm) / self.num_of_val_samples
        print("Validation accuracy: {}".format(acc))

        precision = self.compute_precision(cm)
        print("Validation precision: {}".format(precision))

        recall = self.compute_recall(cm)
        print("Validation recall: {}".format(recall))

        print("\n")
        class_accuracies = self.compute_class_accuracies(cm)
        print("class accuracies: {}".format(class_accuracies))
        # Tuned

        if self.add_tuned:
            print("\n")
            tuned_preds = np.zeros(Y_pred.shape)

            tuned_preds[Y_pred >= 0.35] = 1
            tuned_preds[Y_pred < 0.35] = 0

            tuned_preds = self.convert(tuned_preds)
            tuned_cm = confusion_matrix(self.Y, tuned_preds)

            print("tuned_cm: {}".format(tuned_cm))

            tuned_acc = np.trace(tuned_cm) / self.num_of_val_samples
            print("tuned validation accuracy: {}".format(tuned_acc))

            tuned_precision = self.compute_precision(tuned_cm)
            print("tuned validation precision: {}".format(tuned_precision))

            tuned_recall = self.compute_recall(tuned_cm)
            print("tuned validation recall: {}".format(tuned_recall))

        self.early_stopping_info(self.highest_acc, acc)
        
        if acc > self.highest_acc:
            self.confusion_matrix = cm
            self.highest_acc = acc
            if self.add_tuned:
                self.tuned_confusion_matrix = tuned_cm
            if self.save:
                self.highest_acc_preds = Y_pred
        
        lr = K.eval(self.model.optimizer.lr)
        print("\n")
        print("lr")
        print(lr)

        with self.train_writer.as_default():
            tf.summary.scalar("Accuracy", training_acc, step=epoch)

        with self.validation_writer.as_default():
            tf.summary.scalar("Accuracy", acc, step=epoch)
            tf.summary.scalar("Precision", precision, step=epoch)
            tf.summary.scalar("Recall", recall, step=epoch)
            if self.add_tuned:
                tf.summary.scalar("Tuned Accuracy", tuned_acc, step=epoch)
                tf.summary.scalar("Tuned Precision",
                                  tuned_precision, step=epoch)
                tf.summary.scalar("Tuned Recall", tuned_recall, step=epoch)
            tf.summary.scalar("Learning rate", lr, step=epoch)

        with self.none_writer.as_default():
            tf.summary.scalar("Class Accuracy",
                              class_accuracies[0], step=epoch)
        with self.crackles_writer.as_default():
            tf.summary.scalar("Class Accuracy",
                              class_accuracies[1], step=epoch)
        with self.wheezes_writer.as_default():
            tf.summary.scalar("Class Accuracy",
                              class_accuracies[2], step=epoch)
        with self.both_writer.as_default():
            tf.summary.scalar("Class Accuracy",
                              class_accuracies[3], step=epoch)

    def compute_recall(self, cm):
        precisions = []
        for i, row in enumerate(cm):
            if sum(row) == 0:
                precisions.append(0)
            else:
                precisions.append(row[i] / sum(row))
        return sum(precisions) / len(precisions)

    def compute_precision(self, cm):
        recalls = []
        for i, row in enumerate(cm.T):
            if sum(row) == 0:
                recalls.append(0)
            else:
                recalls.append(row[i] / sum(row))
        return sum(recalls) / len(recalls)

    def compute_class_accuracies(self, cm):
        accs = []
        for i, row in enumerate(cm):
            accs.append(row[i] / sum(row))
        return accs
    
    def collect_samples(self):
        class_start_indexes = [self.Y.index(0), self.Y.index(1), self.Y.index(2), self.Y.index(3)]
        for start_ind in class_start_indexes:
            for i in range(start_ind, start_ind + 10):
                self.samples_to_be_saved.append(
                    (self.highest_acc_preds[i], multi_hot_encoding(self.Y[i]), i))

    def early_stopping_info(self, highest_accuracy, current_accuracy):
        diff = current_accuracy - self.tracker_accuracy
        if diff > self.min_delta:  # 1%
            self.tracker = 0
            self.tracker_accuracy = current_accuracy
        else:
            self.tracker += 1
            if self.tracker == self.es_patience:
                print("The number of epochs since last 1% equals the patience")
                self.model.stop_training = True
            else:
                print("The validation accuracy hasn't increased by 1% in {} epochs".format(
                    self.tracker))
        return None

    def save_png_and_wav(self):
        for i, sample in enumerate(self.samples_to_be_saved):
            diff = abs(sample[0][0] - sample[1][0]) + \
                abs(sample[0][1] - sample[1][1])
            index = sample[2]
            if diff > 0.5:
                print("index: {}, prediction: {}, label: {}".format(
                    index, sample[0], sample[1]))
                fig = plt.figure(figsize=(20, 10))
                spec = self.X[index][:, :, 0]
                display.specshow(
                    spec,
                    y_axis="log",
                    sr=self.sr,
                    cmap="coolwarm"
                )
                plt.show()
                fig.savefig('temp.png')

                # add some assertions?
                sf.write('{}.wav'.format(
                    index), self.validation_data[index][3], self.sr, subtype='PCM_24')

                with file_io.FileIO("temp.png", mode="rb") as input_f:
                    with file_io.FileIO(job_dir + "/images_and_sounds/" + "{}.png".format(index), mode="w+") as output_f:
                        output_f.write(input_f.read())

                with file_io.FileIO("{}.wav".format(index), mode="rb") as input_f:
                    with file_io.FileIO(job_dir + "/images_and_sounds/" + "{}.wav".format(index), mode="w+") as output_f:
                        output_f.write(input_f.read())

    def plot_confusion_matrix(self, cm, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
        """

        figure = plt.figure(figsize=(12, 12))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # print("nN threshold")
        # print(cm.max() / 2.0)

        # Normalize the confusion matrix.
        cm = np.around(cm.astype("float") / cm.sum(axis=1)
                       [:, np.newaxis], decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.0
        # print("threshold")
        # print(threshold)

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            # color = "white" if cm[i, j] > threshold else "black"
            color = "red"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        return figure

    def plot_to_image(self, figure):
        """
        Returns a PNG image that fits the tensorboard requirements.

        Args:
        figure: matplotlib figure containing the plotted conf matrix.
        """
        buf = io.BytesIO()

        # Use plt.savefig to save the plot to a PNG in memory.
        plt.savefig(buf, format="png")

        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)

        # Use tf.image.decode_png to convert the PNG buffer
        # to a TF image. Make sure you use 4 channels.
        image = tf.image.decode_png(buf.getvalue(), channels=4)

        # Use tf.expand_dims to add the batch dimension
        image = tf.expand_dims(image, 0)

        return image

    def prepare_data(self, data):
        """
        Returns the validation data in the right format for prediction.

        Args:
        data: test data to be used in formal (no of elements in total, height, width, channels)
        """
        final_shape = (self.num_of_val_samples,
                       self.shape[0], self.shape[1], self.shape[2])
        X = np.zeros(final_shape, dtype=np.float32,)
        for i, wav in enumerate(data):
            wav = wav[0]
            wav = np.expand_dims(wav, axis=-1)
            X[i, :, :] = wav
        return X

    def convert(self, inp):
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


def info(train_data, val_data=None):
    print("Train: ")
    print_sample_count(train_data[0], train_data[1],
                       train_data[2], train_data[3])
    if val_data:
        print("\nTest: ")
        print_sample_count(val_data[0], val_data[1],
                           val_data[2], val_data[3])
    print("-----------------------")

    ##################


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
        "EPSILON": epsilon
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
    print("Early Stopping Patience: {}".format(params["ES_PATIENCE"]))
    print("-----------------------")

    # data retrieval
    print("Collecting data...")
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

    info([none_train, c_train, w_train, c_w_train],
         [none_test, c_test, w_test, c_w_test])

    # generators
    if bool(params["OVERSAMPLE"]):
        print("Oversampling the data...")
        oversampled_data = oversample_dataset([none_train, c_train, w_train,
                                               c_w_train])
        # print(len(oversampled_data))
        # tg = data_generator(oversampled_data, sr,
        #                     n_classes, shape, batch_size,)
        tg = data_generator(oversampled_data, sr,
                            n_classes, shape, batch_size, initial_channels)
    else:
        tg = data_generator(train_data, sr, n_classes,
                            shape, batch_size, initial_channels)

    # callbacks
    save = bool(params["SAVE"])
    add_tuned = bool(params["ADD_TUNED"])
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_path)
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="accuracy", **lr_params
    )
    confusion_m_callback = cm_callback(
        validation_data, shape, n_classes, sr, save, add_tuned, es_patience, min_delta)
    # early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    #     monitor='accuracy', min_delta=min_delta, patience=es_patience)

    # model setting

    model = conv2d(**model_params)

    # weights + model running/saving

    model.summary()

    if bool(params["CLASS_WEIGHTS"]):
        print("Initializing weights...")
        y_train = label_data(validation_data)
        weights = class_weight.compute_class_weight(
            "balanced", np.unique(y_train), y_train)

        weights = {i: weights[i] for i in range(0, len(weights))}
        print("weights = {}".format(weights))
        model.fit(
            tg,
            epochs=n_epochs,
            verbose=2,
            callbacks=[tensorboard_callback,
                       confusion_m_callback,
                       reduce_lr_callback,
                       #    early_stopping_callback
                       ],
            class_weight=weights,
        )
    else:
        print("No class weights initalized. Model is starting to train...")
        model.fit(
            tg,
            epochs=n_epochs,
            verbose=2,
            callbacks=[tensorboard_callback,
                       confusion_m_callback,
                       reduce_lr_callback,
                       #    early_stopping_callback
                       ],
        )

    model.save("model.h5")

    # Save model.h5 on to google storage
    with file_io.FileIO("model.h5", mode="rb") as input_f:
        with file_io.FileIO(job_dir + "/model.h5", mode="w+") as output_f:
            output_f.write(input_f.read())


##################
#################
#################


if __name__ == "__main__":
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
