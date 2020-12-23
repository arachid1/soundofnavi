from .helpers import *
from matplotlib import pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.keras import backend as K
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix
import itertools
import io
import librosa
from librosa import display
import soundfile as sf


class cm_callback(tf.keras.callbacks.LambdaCallback):
    def __init__(
        self, validation_data, shape, n_classes, sr, save, add_tuned, es_patience, min_delta, job_dir
    ):
        self.shape = shape
        self.n_classes = n_classes
        self.validation_data = validation_data
        self.sr = sr
        self.save = save
        self.add_tuned = add_tuned
        self.es_patience = es_patience
        self.min_delta = min_delta
        self.job_dir = job_dir
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
            self.job_dir + "/logs/train_acc")
        self.validation_writer = tf.summary.create_file_writer(
            self.job_dir + "/logs/val_acc")
        self.none_writer = tf.summary.create_file_writer(
            self.job_dir + "/logs/none")
        self.crackles_writer = tf.summary.create_file_writer(
            self.job_dir + "/logs/crackles")
        self.wheezes_writer = tf.summary.create_file_writer(
            self.job_dir + "/logs/wheezes")
        self.both_writer = tf.summary.create_file_writer(
            self.job_dir + "/logs/both")
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
                    with file_io.FileIO(self.job_dir + "/images_and_sounds/" + "{}.png".format(index), mode="w+") as output_f:
                        output_f.write(input_f.read())

                with file_io.FileIO("{}.wav".format(index), mode="rb") as input_f:
                    with file_io.FileIO(self.job_dir + "/images_and_sounds/" + "{}.wav".format(index), mode="w+") as output_f:
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
            wav = np.repeat(wav[..., np.newaxis], self.shape[2], -1)
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

def visualise_inputs(model, dimension, batch_size):
    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functor = K.function([inp, K.symbolic_learning_phase()], outputs )   # evaluation function

    # Testing
    test = np.random.random((batch_size, 50, dimension))[np.newaxis,...]
    print(test.shape)
    layer_outs = functor([test, 1.])
    print(layer_outs)
