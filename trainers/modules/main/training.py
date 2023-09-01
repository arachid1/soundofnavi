from . import parameters as p
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import numpy as np
import io

import time


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(figure)
    buf.seek(0)

    digit = tf.image.decode_png(buf.getvalue(), channels=4)
    digit = tf.expand_dims(digit, 0)

    return digit


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
        self.n_classes = p.n_classes * 2
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
        score = 0.5 * (sensitivity + specificity)
        return accuracy, sensitivity, specificity, score

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


def train_function(
    model, loss_fn, optimizer, train_dataset, val_dataset, train_writer, val_writer
):
    train_length = len(train_dataset)
    best = 0.0
    wait = 0
    tracker = 0

    for epoch in range(p.n_epochs):
        print("\nEpoch {}/{}".format(epoch + 1, p.n_epochs))
        start_time = time.time()
        val_loss = 0.0

        # model.analyze_layers(domain_examples)

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            # loss_value = train_step(x_batch_train, y_batch_train)
            loss_value = model.train_step(x_batch_train, y_batch_train)
            with train_writer.as_default():
                tf.summary.scalar(
                    "train_loss", loss_value, step=epoch * train_length + step
                )

        # Display metrics at the end of each epoch.
        with train_writer.as_default():
            for m in model.compiled_metrics._metrics:
                print("Training {} over epoch: {}".format(m.name, m.result()))
                if m.name == "confusion_matrix_metric":
                    accuracy, sensitivity, specificity, score = m.metrics_result()
                    figure = plot_confusion_matrix(
                        m.result().numpy(), class_names=p.class_names
                    )
                    cm_image = plot_to_image(figure)
                    tf.summary.image("Train CM", cm_image, step=epoch)
                    tf.summary.scalar("train_accuracy", accuracy, step=epoch)
                    tf.summary.scalar("train_sensitivity", sensitivity, step=epoch)
                    tf.summary.scalar("train_specificity", specificity, step=epoch)
                    tf.summary.scalar("train_score", score, step=epoch)
                else:
                    tf.summary.scalar("train_" + m.name, m.result(), step=epoch)
                m.reset_state()

        # Run a validation loop at the end of each epoch.
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            loss_value = model.test_step(x_batch_val, y_batch_val)
            val_loss += loss_value
        val_loss_avg = val_loss / (step + 1)

        with val_writer.as_default():
            tf.summary.scalar("val_loss", val_loss_avg, step=epoch)
            print("Validation loss over epoch: {}".format(val_loss_avg))
            for m in model.compiled_metrics._metrics:
                print("Validation {} over epoch: {}".format(m.name, m.result()))
                if m.name == "confusion_matrix_metric":
                    accuracy, sensitivity, specificity, score = m.metrics_result()
                    figure = plot_confusion_matrix(
                        m.result().numpy(), class_names=p.class_names
                    )
                    cm_image = plot_to_image(figure)
                    tf.summary.image("Val CM", cm_image, step=epoch)
                    tf.summary.scalar("val_accuracy", accuracy, step=epoch)
                    tf.summary.scalar("val_sensitivity", sensitivity, step=epoch)
                    tf.summary.scalar("val_specificity", specificity, step=epoch)
                    tf.summary.scalar("val_score", score, step=epoch)
                else:
                    tf.summary.scalar("val_" + m.name, m.result(), step=epoch)
                m.reset_state()

        wait += 1
        if score > best:
            best = score
            wait = 0
            tracker = 0
            continue
        tracker += 1
        if wait >= p.es_patience:
            print(
                "Training stopped due to unimproved results for {} epochs".format(
                    p.es_patience
                )
            )
            break
        else:
            print(
                "The validation tracker metric at {} hasn't increased  in {} epochs".format(
                    best, tracker
                )
            )
            if (not (tracker == 0)) and tracker % p.lr_patience == 0:
                if optimizer.lr > p.min_lr:
                    optimizer.lr = optimizer.lr * p.factor
                    print("Lr has been adjusted to {}".format(optimizer.lr.numpy()))

        print("Time taken: %.2fs" % (time.time() - start_time))
