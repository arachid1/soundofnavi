from . import parameters
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Accent)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    return figure


class ConfusionMatrixMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the running average of the confusion matrix
    """

    def __init__(self, **kwargs):
        super(ConfusionMatrixMetric, self).__init__(
            name="confusion_matrix_metric", **kwargs
        )  # handles base args (e.g., dtype)
        self.n_classes = parameters.n_classes * 2
        self.total_cm = self.add_weight(
            "total", shape=(self.n_classes, self.n_classes), initializer="zeros"
        )

    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total_cm.assign_add(self.confusion_matrix(y_true, y_pred))
        return self.total_cm

    def result(self):
        return self.total_cm

    def metrics_result(self):
        accuracy = tf.linalg.trace(self.total_cm) / tf.reduce_sum(self.total_cm)
        sensitivity = self.total_cm[1, 1] + self.total_cm[2, 2] + self.total_cm[3, 3]
        sensitivity = sensitivity / (
            sum(self.total_cm[1, :])
            + sum(self.total_cm[2, :])
            + sum(self.total_cm[3, :])
        )
        specificity = self.total_cm[0, 0] / sum(self.total_cm[0, :])
        icbhi_score = 0.5 * (sensitivity + specificity)
        return accuracy, sensitivity, specificity, icbhi_score

    def confusion_matrix(self, y_true, y_pred):
        """
        Make a confusion matrix
        """
        y_true = tf.map_fn(lambda _y: tf.gather(_y, 1) + tf.math.reduce_sum(_y), y_true)
        y_pred = tf.math.round(y_pred)
        y_pred = tf.map_fn(lambda _y: tf.gather(_y, 1) + tf.math.reduce_sum(_y), y_pred)
        cm = tf.math.confusion_matrix(
            y_true, y_pred, dtype=tf.float32, num_classes=self.n_classes
        )
        return cm
