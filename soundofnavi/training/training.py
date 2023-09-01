from soundofnavi.main import parameters as p
from soundofnavi.metrics.confusion_matrix import ConfusionMatrixMetric, plot_to_image, plot_confusion_matrix
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import numpy as np
import io
import time



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
