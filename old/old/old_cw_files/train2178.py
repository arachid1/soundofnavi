from .modules.helpers import *
from .modules.generators import *
from .modules.metrics import *
from .modules.callbacks import *
from .modules.pneumonia import *
from .modules.parse_functions import *
from .modules.augmentation import *
from .core import *

import argparse
import pickle
import tensorflow as tf
import tensorflow_addons
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
import glob
import wandb
from wandb.keras import WandbCallback

class metric_callback_cust(tf.keras.callbacks.LambdaCallback):

    def __init__(self, validation_data, shape, n_classes, job_dir, min_delta, es_patience, lr_patience, min_lr, factor, epoch_start, target, split_count, average_mode='binary'):

        self.validation_data = validation_data
        self.shape = shape
        self.n_classes = n_classes
        self.job_dir = job_dir
        self.min_delta = min_delta
        self.es_patience = es_patience
        self.lr_patience = lr_patience
        self.min_lr = min_lr
        self.factor = factor
        self.epoch_start = epoch_start
        self.target = target
        self.average_mode = average_mode
        self.split_count = split_count
        self.best_f1 = 0
        self.best_cm = 0
        self.best_auc = 0
        self.best_accuracy = 0
        self.best_precision = 0
        self.best_recall = 0
        self.best_epoch = None
        self.tracker = 0
        self.tracker_metric = 0
    
    # def on_train_begin(self, logs=None):

    def on_train_end(self, logs=None):
        tf.print("\nThe metrics for the epoch with the best target metric are: ")
        tf.print("Best cm: {}".format(self.best_cm))
        tf.print("Best f1: {}".format(self.best_f1))
        tf.print("Best accuracy: {}".format(self.best_accuracy))
        tf.print("Best recall: {}".format(self.best_recall))
        tf.print("Best precision: {}".format(self.best_precision))
        tf.print("Best AUC: {}".format(self.best_auc))
        tf.print("Best epoch: {}".format(self.best_epoch))

        wandb.run.summary.update({"best_val_f1": self.best_f1})
        wandb.run.summary.update({"best_val_accuracy": self.best_accuracy})
        wandb.run.summary.update({"best_val_precision": self.best_precision})
        wandb.run.summary.update({"best_val_recall": self.best_recall})
        wandb.run.summary.update({"best_auc": self.best_auc})
        wandb.run.summary.update({"best_epoch": self.best_epoch})

    def on_epoch_end(self, epoch, logs=None):
        # print(logs)

        one_indexed_epoch = epoch + 1

        cm = np.zeros((self.n_classes + 1, self.n_classes + 1))

        y_true_all = []
        y_pred_all = []
        preds_all = []

        for batch, y_true in self.validation_data:

            preds = self.model.predict(batch)
            preds = preds.reshape(len(preds))
            y_pred = np.zeros(preds.shape)
            y_pred[preds >= 0.5] = 1
            y_pred[preds < 0.5] = 0
            preds_all.extend(preds)
            y_true_all.extend(y_true.numpy())
            y_pred_all.extend(y_pred)
    
        # AUC & PR Curve
        precisions, recalls, thresholds = precision_recall_curve(y_true_all, preds_all)
        area_under_curve = auc(recalls, precisions, )
        # print(precisions)
        # print(recalls)
        # pr_display = plot_precision_recall_curve(self.model, X_test, y_test)
        plt.xticks(np.arange(0, 1, 0.1))
        plt.yticks(np.arange(0, 1, 0.1))
        plt.close()
        # PrecisionRecallDisplay.plot()
        pr_display = PrecisionRecallDisplay(precisions, recalls, average_precision=0.0, estimator_name=None)
        pr_display = pr_display.plot()
        pr_display.ax_.get_legend().remove()
        pr_display.ax_.xaxis.set_ticks(np.arange(0, 1, 0.1))
        pr_display.ax_.yaxis.set_ticks(np.arange(0, 1, 0.1))
        path = "pr_curves/test_pr_curve_{}.png".format(one_indexed_epoch)
        plt.savefig(path)

        # CM, accuracy, precision, recall, f1 and class accuracies
        cm = confusion_matrix(y_true_all, y_pred_all)
        cm_sum = np.sum(cm)
        acc = np.trace(cm) / cm_sum
        class_accuracies=self.compute_class_accuracies(cm)

        precision, recall, f1, support = precision_recall_fscore_support(y_true_all, y_pred_all, average=self.average_mode)
        
        tf.print("Confusion matrix: \n {}".format(cm))
        tf.print("Validation f1: {:.2f}".format(f1*100))
        tf.print("Validation accuracy: {:.2f}".format(acc*100))
        tf.print("Validation recall: {:.2f}".format(recall*100))
        tf.print("Validation precision: {:.2f}".format(precision*100))
        tf.print("Validation AUC: {}".format(area_under_curve))
        tf.print("Validation class accuracies: {}".format(class_accuracies))

        wandb.log({'train_accuracy': logs['accuracy'], 'val_accuracy': logs['val_accuracy']})
        wandb.log({'train_loss': logs['loss'], 'val_loss': logs['val_loss']})
        wandb.log({'val_recall': recall})
        wandb.log({'val_precision': precision})
        wandb.log({'val_f1': f1})
        wandb.log({'val_auc': area_under_curve})
        wandb.log({'lr': self.model.optimizer.lr})

        print("TARGET: {}".format(self.target))
        if self.target == 0: # auc 
            target_metric = area_under_curve
            print("target metric(test auc): {}".format(target_metric))
            best_target_metric = self.best_auc
        elif self.target == 1: # f1
            target_metric = f1
            print("target metric(test f1): {}".format(target_metric))
            best_target_metric = self.best_f1
    
        # column_sums = np.sum(cm, axis=0)
        # print("column_sums: {}".format(column_sums))
        # print("cm_sum: {}".format(cm_sum))
        # if (column_sums[0] >= (0.9 * cm_sum)) or (column_sums[1] >= (0.9 * cm_sum)):
        #     print("The training is defaulting to either class.")
        # else:
        print("inside not defaulting")
        if one_indexed_epoch >= self.epoch_start: 
            self.early_stopping_info(target_metric)
            # self.adaptive_lr_info()
        if one_indexed_epoch >= self.epoch_start and target_metric > best_target_metric:
            "New best results were achieved."
            self.best_f1 = f1
            self.best_cm = cm
            self.best_auc = area_under_curve
            self.best_accuracy = acc
            self.best_precision = precision
            self.best_recall = recall
            self.best_epoch = one_indexed_epoch
            wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=y_true_all, preds=y_pred_all,) })
            wandb.log({"pr_{}".format(epoch) : wandb.plot.pr_curve(y_true_all, preds_all, title="Precision v. Recall at {} epoch".format(one_indexed_epoch))}) #TODO: CAREFUL WITH PIECE OF CODE I CHANGED IN THE PR_CURVE.PY -> probas[:, i] becomes probas
            wandb.log({"pr_curve_{}".format(one_indexed_epoch): wandb.Image(path)})
            print("Saving model...")
            self.model.save(self.job_dir + "/{}/model_epoch{}.h5".format(self.split_count, one_indexed_epoch))
            
    def adaptive_lr_info(self):  
        self.model.optimizer.lr = self.model.optimizer.lr * self.factor
        print("Lr has been adjusted to {}".format(self.model.optimizer.lr.numpy()))    
        # if self.tracker % self.lr_patience == 0:
        #         if self.model.optimizer.lr > self.min_lr:
        #             self.model.optimizer.lr = self.model.optimizer.lr * self.factor
        #             print("Lr has been adjusted to {}".format(self.model.optimizer.lr.numpy()))      
    
    def early_stopping_info(self, current_metric):
        diff = current_metric - self.tracker_metric
        if diff > self.min_delta:
            self.tracker = 0
            self.tracker_metric = current_metric
        else:
            self.tracker += 1
            # if self.tracker % self.lr_patience == 0:
            #     if self.model.optimizer.lr > self.min_lr:
            #         self.model.optimizer.lr = self.model.optimizer.lr * self.factor
            #         print("Lr has been adjusted to {}".format(self.model.optimizer.lr.numpy()))
            if self.tracker == self.es_patience:
                print("The number of epochs since last 1% equals the patience")
                self.model.stop_training = True
            else:
                print("The validation tracker metric at {} hasn't increased by {} in {} epochs".format(self.tracker_metric, self.min_delta, self.tracker))
        return None

    def compute_class_accuracies(self, cm):
        accs = []
        for i, row in enumerate(cm):
            accs.append(row[i] / sum(row))
        return accs

def conv2d(N_CLASSES, SR, BATCH_SIZE, LR, SHAPE, WEIGHT_DECAY, LL2_REG, EPSILON, LABEL_SMOOTHING, OPTIMIZER_CB):

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
        learning_rate=OPTIMIZER_CB, beta_1=0.9, beta_2=0.999, epsilon=EPSILON, decay=WEIGHT_DECAY, amsgrad=False
    )
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING)
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=[
            'accuracy', 
        ],
    )
    return model
    

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

    six = bool(params["SIX"])
    concat = bool(params["CONCAT"])

    epoch_start = int(params["EPOCH_START"])
    target = int(params["TARGET"])

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

    time_masking = int(spec_params["TIME_MASKING"])
    frequency_masking = int(spec_params["FREQUENCY_MASKING"])
    loudness = int(spec_params["LOUDNESS"])

    wav_quantity = int(wav_params['QUANTITY'])
    wav_path = str(wav_params["WAV_PATH"])

    print("Model Parameters: {}".format(model_params))
    print("Learning Rate Parameters: {}".format(lr_params))
    print("Early Stopping Patience and Delta: {}, {}%".format(es_patience, min_delta*100))
    print("Six: {} and Concat: {}".format(six, concat))
    print("Audio augmentations params: {}".format(wav_params))
    print("Spectrogram augmentations params: {}".format(spec_params))
    print("-----------------------")

    train_test_ratio = 0.8

    split_count = 0
    filenames = []
    labels = []

    wandb_name = __file__.split('/')[-1].split('.')[0]
    # print(wandb_name)
    wandb.init(project="tensorboard-integration", name=wandb_name, sync_tensorboard=False)

    config = wandb.config

    config.n_classes = n_classes
    config.n_epochs = n_epochs
    config.sr = sr
    config.lr = lr
    config.batch_size = batch_size
    config.ll2_reg = ll2_reg
    config.weight_decay = weight_decay
    config.label_smoothing = label_smoothing
    config.es_patience = es_patience
    config.min_delta = min_delta
    config.initial_channels = initial_channels
    config.factor = factor
    config.patience = patience
    config.min_lr = min_lr

    train_file_name = train_file.split('/')[-1].split('.')[0]

    dataset_path = '../../data/txt_datasets/{}'.format(train_file_name)

    filenames, labels = process_bangladesh(six, dataset_path)

    samples = list(zip(filenames, labels))

    random.shuffle(samples)

    __, stratify_labels = zip(*samples)

    grouped_val_samples, grouped_train_samples = train_test_split(
            samples, test_size=train_test_ratio, stratify=stratify_labels)

    random.shuffle(grouped_val_samples)
    random.shuffle(grouped_train_samples)

    if concat:
        val_dataset, train_dataset = process_data(grouped_val_samples, grouped_train_samples, concatenate_specs, shape, batch_size, initial_channels)
    else:
        train_samples = [[recording, s[1]] for s in grouped_train_samples for recording in s[0]]
        print(len(train_samples))

        if bool(wav_params['ADD']):
            print("Augmenting audios...")
            augmented_indices = random.sample(range(len(train_samples)), wav_quantity)
            augmented_audios = [train_samples[i] for i in augmented_indices]
            for i in range(len(augmented_audios)):
                augmented_audios[i][0] = augmented_audios[i][0].replace(params["PARAM"], wav_path) # TODO: rename some variables
            train_samples += augmented_audios
            print("Length of training samples after augmentation: {}".format(len(train_samples)))

        
        if bool(spec_params['ADD']):
            print("Augmenting spectrograms...")
            augmented_indices = random.sample(range(len(train_samples)), int(spec_params['QUANTITY']))
            augmented_specs = [train_samples[i] for i in augmented_indices] 
            augmented_specs = apply_augmentations(augmented_specs, os.path.join(dataset_path, "augmented"), len(augmented_specs), time_masking, frequency_masking, loudness)
            train_samples += augmented_specs
            print("Length of training samples after augmentation: {}".format(len(train_samples)))

        val_samples = [[recording, s[1]] for s in grouped_val_samples for recording in s[0]]
        # print(len(val_samples))
        # print(len(val_samples))
        print("With concat = {} and six = {}, size of training set: {}...".format(concat, six, len(train_samples)))
        print("...and size of validation set: {}".format(len(val_samples)))
        val_dataset, train_dataset = process_data(val_samples, train_samples, generate_spec, shape, batch_size, initial_channels)
    
    # weights
    weights = None

    if bool(params["CLASS_WEIGHTS"]):
        print("Initializing weights...")
        weights = class_weight.compute_class_weight(
            "balanced", np.unique(labels), labels)
        weights = {i: weights[i] for i in range(0, len(weights))}
        print("weights = {}".format(weights))
    
    # callbacks
    cyclical_lr_callback = tensorflow_addons.optimizers.CyclicalLearningRate(
        initial_learning_rate=min_lr,
        maximal_learning_rate=lr,
        step_size=1315, # nb_iterations = 263 -> [526, 2630] according to medium article/Leslie smith
        scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
        scale_mode='cycle')
    
    model_params['OPTIMIZER_CB'] = cyclical_lr_callback

    metrics_callback = metric_callback_cust(val_dataset, shape, n_classes, job_dir, min_delta, es_patience, patience, min_lr, factor, epoch_start, target, split_count)

    gpus = tf.config.experimental.list_logical_devices('GPU')

    # model setting
    model = conv2d(**model_params)

    model.summary()

    if gpus:
        for gpu in gpus:
            model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=n_epochs,
                verbose=2,
                class_weight=weights,
                callbacks=[metrics_callback]
            )

    model.save(job_dir + "/{}/model_{}.h5".format(split_count, n_epochs))

    print("\nStarting the Bangladesh analysis...")
    models = sorted(glob.glob(os.path.join(job_dir, "{}/*.h5".format(split_count))))
    best_bd_target_metric = 0
    thresholds = [1, 2, 3]
    for model_path in models:
        for pneumonia_threshold in thresholds:
            # print(model_path)
            model = conv2d(**model_params)
            model.load_weights(model_path)
            # print(model)
            epoch = model_path.split('/')[-1].split('.')[0].split('_')[-1]
            print("Epoch: {}".format(epoch))
            excel_dest = job_dir + "/validation_sheet_{}.xls".format(epoch)
            bd_f1, bd_accuracy, bd_precision, bd_recall, bd_auc, bd_y_true, bd_y_pred = generate_bangladesh_sheet(model, grouped_val_samples, excel_dest, six, initial_channels, pneumonia_threshold)
            if target == 0: # auc 
                target_metric = bd_auc
            elif target == 1: # f1
                target_metric = bd_f1
            if target_metric > best_bd_target_metric:
                best_bd_target_metric = target_metric
                wandb.run.summary.update({"best_bd_f1": bd_f1})
                wandb.run.summary.update({"best_bd_auc": bd_auc})
                wandb.run.summary.update({"best_bd_accuracy": bd_accuracy})
                wandb.run.summary.update({"best_bd_precision": bd_precision})
                wandb.run.summary.update({"best_bd_recall": bd_recall})
                wandb.run.summary.update({"best_bd_epoch": epoch})
                wandb.log({"bd_conf_mat" : wandb.plot.confusion_matrix(probs=None,
                                y_true=bd_y_true, preds=bd_y_pred,) })
                # wandb.run.summary.update({"best_val_accuracy": self.best_accuracy})
                # wandb.run.summary.update({"best_val_precision": self.best_precision})
                # wandb.run.summary.update({"best_val_recall": self.best_recall})
                # wandb.run.summary.update({"best_auc": self.best_auc})
                # wandb.run.summary.update({"best_epoch": self.best_epoch})

if __name__ == "__main__":
    print("Tensorflow Version: {}".format(tf.__version__))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
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
    parser.add_argument(
        "--wav-params",
        help="Augmentation parameters for audio files",
        required=True,
    )
    parser.add_argument(
        "--spec-params",
        help="Augmentation parameters for spectrogram",
        required=True,
    )
    args = parser.parse_args()
    arguments = args.__dict__
    job_dir = arguments.pop("job_dir")
    params = arguments.pop("params")
    params = json.loads(params)
    wav_params = arguments.pop("wav_params")
    wav_params = json.loads(wav_params)
    spec_params = arguments.pop("spec_params")
    spec_params = json.loads(spec_params)
    train_model(**arguments)
