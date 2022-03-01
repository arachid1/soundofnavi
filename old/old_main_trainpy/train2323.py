from .modules.helpers import *
from .modules.generators import *
from .modules.metrics import *
from .modules.callbacks import *
from .modules.pneumonia import *
from .modules.parse_functions import *
from .modules.augmentation import *
from .core import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import argparse
import pickle
import tensorflow as tf
import tensorflow_addons
import sys
import shutil
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

class custom_metric_callback(tf.keras.callbacks.LambdaCallback):

    def __init__(self, validation_data, shape, initial_channels, n_classes, job_dir, sr, min_delta, es_patience, lr_patience, min_lr, factor, epoch_start, target, job_count, clause, samples_to_vis, parse_function, cuberooting, normalizing, adaptive_lr=True, average_mode='binary', clause_portion=0.9):

        self.validation_data = validation_data
        self.shape = shape
        self.initial_channels = initial_channels
        self.n_classes = n_classes
        self.job_dir = job_dir
        self.sr = sr
        self.min_delta = min_delta
        self.es_patience = es_patience
        self.lr_patience = lr_patience
        self.min_lr = min_lr
        self.factor = factor
        self.epoch_start = epoch_start
        self.target = target
        self.average_mode = average_mode
        self.job_count = job_count
        self.clause = clause
        self.adaptive_lr = adaptive_lr
        self.clause_portion = clause_portion
        self.samples_to_vis = samples_to_vis
        self.parse_function = parse_function
        self.cuberooting = cuberooting
        self.normalizing = normalizing
        self.best_model = None
        self.job_directory = None
        self.best_f1 = 0
        self.best_cm = 0
        self.best_normalized_cm = 0
        self.best_auc = 0
        self.best_accuracy = 0
        self.best_precision = 0
        self.best_recall = 0
        self.best_class_accuracies = None
        self.best_epoch = None
        self.tracker = 0
        self.tracker_metric = 0
    
    def on_train_begin(self, logs=None):
        if not (os.path.exists(self.job_dir)):
            os.mkdir(self.job_dir)
        self.job_directory = os.path.join(self.job_dir, str(self.job_count))
        if not (os.path.exists(self.job_directory)):
            os.mkdir(self.job_directory)

    def on_train_end(self, logs=None):
        tf.print("\nThe metrics for job {} with the best target metric are: ".format(self.job_count))
        tf.print("Best cm: {}".format(self.best_cm))
        tf.print("Best normalized cm: {}".format(self.best_normalized_cm))
        tf.print("Best f1: {:.2f}".format(self.best_f1))
        tf.print("Best accuracy: {:.2f}".format(self.best_accuracy))
        tf.print("Best recall: {:.2f}".format(self.best_recall))
        tf.print("Best precision: {:.2f}".format(self.best_precision))
        tf.print("Best AUC: {:.2f}".format(self.best_auc))
        tf.print("Best class accuracies: {}".format(self.best_class_accuracies))
        tf.print("Best epoch: {}".format(self.best_epoch))

        wandb.run.summary.update({"best_val_f1": self.best_f1})
        wandb.run.summary.update({"best_val_accuracy": self.best_accuracy})
        wandb.run.summary.update({"best_val_precision": self.best_precision})
        wandb.run.summary.update({"best_val_recall": self.best_recall})
        wandb.run.summary.update({"best_auc": self.best_auc})
        wandb.run.summary.update({"best_epoch": self.best_epoch})
        # if len(self.shape) == 2:
        #     for filename, label in self.samples_to_vis:
        #         spec, ___ = self.parse_function(filename, label, self.shape, self.initial_channels, self.cuberooting, self.normalizing)
        #         spec = spec.numpy()
        #         spec_to_vis = np.squeeze(spec)
        #         spec_to_predict = np.expand_dims(spec, axis=0)
        #         output = self.best_model.predict(spec_to_predict)
        #         rounded_output = round(output[0][0])
        #         name = filename.split('.txt')[0].split('/')[-1]
        #         path = os.path.join(self.job_dir, str(self.job_count))
        #         name = os.path.join(path, name + str('_label_{}'.format(int(label))) + str('_pred_{}'.format(rounded_output)) + str('.png'))
        #         print(name + str(", {}".format(output)))
        #         visualize_spectrogram(spec_to_vis, self.sr, name)


    def on_epoch_end(self, epoch, logs=None):
        # print(logs)

        one_indexed_epoch = epoch + 1

        cm = np.zeros((self.n_classes + 1, self.n_classes + 1))

        y_true_all = []
        y_pred_all = []
        preds_all = []

        fns = []
        fps = []
        tps = []

        for batch, y_true, filename in self.validation_data:
            batch = batch.numpy()
            y_true = y_true.numpy()[0]
            filename = str(filename.numpy()[0])
            preds = self.model.predict(batch)
            preds = preds.reshape(len(preds))
            y_pred = np.zeros(preds.shape)
            y_pred[preds >= 0.5] = 1
            y_pred[preds < 0.5] = 0
            spec_name = filename.split('/')[-1].split('.')[0]
            batch = np.squeeze(batch)
            if y_true == 1 and y_pred == 0: # fn
                dest = os.path.join(self.job_directory, "fn", spec_name)
                fns.append([batch, dest])
            if y_true == 0 and y_pred == 1: # fp
                dest = os.path.join(self.job_directory, "fp", spec_name)
                fps.append([batch, dest])
            if y_true == 1 and y_pred == 1: # tp
                dest = os.path.join(self.job_directory, "tp", spec_name)
                tps.append([batch, dest])
            preds_all.extend(preds)
            y_true_all.append(y_true)
            y_pred_all.append(y_pred)
    
        # AUC & PR Curve
        # precisions, recalls, thresholds = precision_recall_curve(y_true_all, preds_all)
        # area_under_curve = auc(recalls, precisions, )

        # pr_display = plot_precision_recall_curve(self.model, X_test, y_test)
        # plt.xticks(np.arange(0, 1, 0.1))
        # plt.yticks(np.arange(0, 1, 0.1))
        # plt.close()
        # PrecisionRecallDisplay.plot()
        # pr_display = PrecisionRecallDisplay(precisions, recalls, average_precision=0.0, estimator_name=None)
        # pr_display = pr_display.plot()
        # pr_display.ax_.get_legend().remove()
        # pr_display.ax_.xaxis.set_ticks(np.arange(0, 1, 0.1))
        # pr_display.ax_.yaxis.set_ticks(np.arange(0, 1, 0.1))
        # path = "pr_curves/test_pr_curve_{}.png".format(one_indexed_epoch)
        # plt.savefig(path)

        # CM, accuracy, precision, recall, f1 and class accuracies
        cm = confusion_matrix(y_true_all, y_pred_all)
        normalized_cm = confusion_matrix(y_true_all, y_pred_all, normalize='true')
        cm_sum = np.sum(cm)
        acc = np.trace(cm) / cm_sum
        class_accuracies=self.compute_class_accuracies(cm)

        precision, recall, f1, support = precision_recall_fscore_support(y_true_all, y_pred_all, average=self.average_mode)
        
        tf.print("Confusion matrix: \n {}".format(cm))
        tf.print("Normalized Confusion matrix: \n {}".format(normalized_cm))
        tf.print("Validation f1: {:.2f}".format(f1*100))
        tf.print("Validation accuracy: {:.2f}".format(acc*100))
        tf.print("Validation recall: {:.2f}".format(recall*100))
        tf.print("Validation precision: {:.2f}".format(precision*100))
        # tf.print("Validation AUC: {}".format(area_under_curve))
        tf.print("Validation class accuracies: {}".format(class_accuracies))

        wandb.log({'train_accuracy': logs['accuracy']})
        wandb.log({'train_loss': logs['loss']})
        wandb.log({'val_recall': recall})
        wandb.log({'val_precision': precision})
        wandb.log({'val_f1': f1})
        # wandb.log({'val_auc': area_under_curve})
        wandb.log({'lr': self.model.optimizer.lr})

        # print("TARGET: {}".format(self.target))
        if self.target == 0: # auc 
            target_metric = area_under_curve
            # print("target metric(test auc): {}".format(target_metric))
            best_target_metric = self.best_auc
        elif self.target == 1: # f1
            target_metric = f1
            # print("target metric(test f1): {}".format(target_metric))
            best_target_metric = self.best_f1
    
        column_sums = np.sum(cm, axis=0)
        # print("column_sums: {}".format(column_sums))
        # print("cm_sum: {}".format(cm_sum))
        if self.clause:
            print("The clause is activated.")
            if (column_sums[0] >= (self.clause_portion * cm_sum)) or (column_sums[1] >= (self.clause_portion * cm_sum)) or (column_sums[1] >= (0.6 * cm_sum)):
                print("The training is defaulting to either class.")
                return
            else:
                print("Not defaulting.")
        if one_indexed_epoch >= self.epoch_start: 
            self.early_stopping_info(target_metric)
            if self.adaptive_lr:
                self.adaptive_lr_info()
        if one_indexed_epoch >= self.epoch_start and target_metric > best_target_metric:
            "New best results were achieved."
            self.best_model = self.model
            self.best_f1 = f1
            self.best_cm = cm
            self.best_normalized_cm = normalized_cm
            # self.best_auc = area_under_curve
            self.best_accuracy = acc
            self.best_precision = precision
            self.best_recall = recall
            self.best_class_accuracies = class_accuracies
            self.best_epoch = one_indexed_epoch
            # wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
            #             y_true=y_true_all, preds=y_pred_all,) })
            # wandb.log({"pr_{}".format(epoch) : wandb.plot.pr_curve(y_true_all, preds_all, title="Precision v. Recall at {} epoch".format(one_indexed_epoch))}) #TODO: CAREFUL WITH PIECE OF CODE I CHANGED IN THE PR_CURVE.PY -> probas[:, i] becomes probas
            # wandb.log({"pr_curve_{}".format(one_indexed_epoch): wandb.Image(path)})
            print("Saving model...")
            self.model.save(self.job_dir + "/{}/model_epoch{}.h5".format(self.job_count, one_indexed_epoch))
            if os.path.exists(os.path.join(self.job_directory, "fp")):
                shutil.rmtree(os.path.join(self.job_directory, "fp"))
                shutil.rmtree(os.path.join(self.job_directory, "fn"))
                shutil.rmtree(os.path.join(self.job_directory, "tp"))
            os.mkdir(os.path.join(self.job_directory, "fp"))
            os.mkdir(os.path.join(self.job_directory, "fn"))
            os.mkdir(os.path.join(self.job_directory, "tp"))
            for fn in fns:
                visualize_spectrogram(fn[0], 8000, fn[1])
            for fp in fps:
                visualize_spectrogram(fp[0], 8000, fp[1])
            for tp in tps:
                visualize_spectrogram(tp[0], 8000, tp[1])
            
    def adaptive_lr_info(self):  
        if (not (self.tracker == 0)) and self.tracker % self.lr_patience == 0:
                if self.model.optimizer.lr > self.min_lr:
                    self.model.optimizer.lr = self.model.optimizer.lr * self.factor
                    print("Lr has been adjusted to {}".format(self.model.optimizer.lr.numpy()))      
    
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
            # if (not (self.tracker == 0)):
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

def custom_generate_spec(filename, label, shape, initial_channels, cuberooting, normalizing):

    spectrogram = tf.io.read_file(filename)
    spectrogram = tf.strings.split(spectrogram)
    spectrogram = tf.strings.split(spectrogram, sep=',')
    spectrogram = tf.strings.to_number(spectrogram)
    spectrogram = spectrogram.to_tensor()
    spectrogram = tf.reshape(spectrogram, shape)
    if cuberooting:
        spectrogram = tf.math.pow(spectrogram, 1/3)
    if normalizing:
        max_value = tf.reduce_max(spectrogram)
        spectrogram = tf.math.divide(spectrogram, max_value)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    spectrogram = tf.tile(spectrogram, [1, 1, initial_channels])
    return spectrogram, tf.cast(label, tf.float32), filename

def mixednet(N_CLASSES, SR, BATCH_SIZE, LR, SHAPE, INITIAL_CHANNELS, WEIGHT_DECAY, LL2_REG, EPSILON, LABEL_SMOOTHING):

    KERNELS = [3, 4, 5, 6, 7]
    POOL_SIZE=2
    i = layers.Input(shape=SHAPE + (INITIAL_CHANNELS,))
    x = layers.Conv2D(8, (3, 3), padding='same', activation='relu')(i)
    x = MixDepthGroupConvolution2D(kernels=KERNELS)(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = MixDepthGroupConvolution2D(kernels=KERNELS)(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MixDepthGroupConvolution2D(kernels=KERNELS)(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MixDepthGroupConvolution2D(kernels=KERNELS)(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MixDepthGroupConvolution2D(kernels=KERNELS)(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    o = layers.Dense(N_CLASSES, activity_regularizer=l2(
        LL2_REG), activation="sigmoid")(x)
        # delete above

    model = Model(inputs=i, outputs=o, name="conv2d")
    opt = tf.keras.optimizers.Adam(
        lr=LR, beta_1=0.9, beta_2=0.999, epsilon=EPSILON, decay=WEIGHT_DECAY, amsgrad=False
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

@tf.function
def custom_mix_up(ds_one, ds_two, batch_size, alpha=0.2):

    # Unpack two datasets
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    labels_one = tf.cast(labels_one, tf.float32)
    labels_two = tf.cast(labels_two, tf.float32)

    # Sample lambda and reshape it to do the mixup
    l = tf.random.uniform(shape=[], minval=0.3, maxval=0.7)
    # tf.print(l)

    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))
    x_l_2 = 1 - x_l

    images_one_mod = images_one * x_l
    images_two_mod = images_two * x_l_2
    
    images = images_one_mod + images_two_mod

    max_value = tf.reduce_max(images)
    images = tf.math.divide(images, max_value)

    labels = (labels_one * y_l) + (labels_two * (1 - y_l))
    images = tf.squeeze(images, axis=0)
    labels = tf.squeeze(labels, axis=0)

    return images, labels[0]

def normalize(spec):
  """Normalize input image channel-wise to zero mean and unit variance."""
  return spec / np.max(spec)

def augment_and_mix(path, job_version, wav_path, shape, augm_path, severity=3, width=3, depth=-1, alpha=1.):
  """Perform AugMix augmentations and compute mixture.
  Args:
    image: Raw input image as float32 np.ndarray of shape (h, w, c)
    severity: Severity of underlying augmentation operators (between 1 to 10).
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
      from [1, 3]
    alpha: Probability coefficient for Beta and Dirichlet distributions.
  Returns:
    mixed: Augmented and mixed image.
  """
  ws = np.float32(
      np.random.dirichlet([alpha] * width))
  m = np.float32(np.random.beta(alpha, alpha))
  image = load_text(path)
#   visualize_spectrogram(image, 8000, 'augmix/orig.png')

  mix = np.zeros(shape, dtype=np.float32)
  for i in range(width):
    d = depth if depth > 0 else np.random.randint(1, 4)
    for i in range(d):
        wav_folder = wav_path[i]
        new_path = path.replace(job_version, wav_folder) 
        if wav_folder == 'v42':
            new_path = new_path.replace('v44', 'v46')
        if wav_folder == 'v43':
            new_path = new_path.replace('v44', 'v45')
        if wav_folder == 'v48':
            new_path = new_path.replace('v44', 'v47')
        image_aug = load_text(new_path)
    # Preprocessing commutes since all coefficients are convex
    image_aug = normalize(image_aug)
    mix += (ws[i] * image_aug)
#   visualize_spectrogram(mix, 8000, 'augmix/mix.png')
  mixed = (1 - m) * normalize(image) + m * mix
  file_name = path.split('/')[-1].split('.')[0]
#   destination = os.path.join(augm_path, file_name)
  file_path = write_to_txt_file(mixed, augm_path, file_name)
#   visualize_spectrogram(mixed, 8000, 'augmix/mixed.png')
  return file_path

def apply_augmix_augmentation(train_samples, augmix_quantity, dataset_path, job_version, wav_path, label, shape):

    augmented_indices = list(range(0, len(train_samples)))
    random.shuffle(augmented_indices)
    augmented_audios = [train_samples[i] for i in augmented_indices if (train_samples[i][1] == label)]
    random.shuffle(augmented_audios)
    augmented_audios = augmented_audios[:augmix_quantity]

    folder_id = 1
    subfolder = 'augmented_mix'
    augm_path = os.path.join(dataset_path, subfolder)
    while os.path.exists(augm_path):
        subfolder = 'augmented_mix_{}'.format(folder_id)
        augm_path = os.path.join(dataset_path, subfolder)
        folder_id += 1
    print("The path used for augmix augmentations is {}".format(augm_path))

    os.mkdir(augm_path)

    augmented_samples = []
    for sample, label in augmented_audios:
        sample = augment_and_mix(sample, job_version, wav_path, shape, augm_path, depth=len(wav_path))
        augmented_samples.append([sample, label])
    
    return augmented_samples

def train_model(train_file, job_dir, params, wav_params, spec_params, model_to_be_trained, job_count, pneumonia_augmix_quantity, non_pneumonia_augmix_quantity, repeat_factor):
    logs_path = job_dir + "/logs/"

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

    job_version = params["PARAM"]

    es_patience = params["ES_PATIENCE"]
    min_delta = params["MIN_DELTA"]

    six = bool(params["SIX"])
    concat = bool(params["CONCAT"])

    epoch_start = int(params["EPOCH_START"])
    target = int(params["TARGET"])
    clause = int(params["CLAUSE"])
    testing = int(params["TESTING"])
    adaptive_lr = int(params["ADAPTIVE_LR"])
    cuberooting = int(params["CUBEROOTING"])
    normalizing = int(params["NORMALIZING"])

    initial_channels = params["INITIAL_CHANNELS"]
    
    model_params = {
        "N_CLASSES": n_classes,
        "SR": sr,
        "BATCH_SIZE": batch_size,
        "LR": lr,
        "SHAPE": shape,
        "INITIAL_CHANNELS": initial_channels,
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

    augmentation = bool(params['AUGMENTATION'])
    spec_add = bool(spec_params['ADD'])
    spec_quantity = int(spec_params["QUANTITY"])
    time_masking = int(spec_params["TIME_MASKING"])
    frequency_masking = int(spec_params["FREQUENCY_MASKING"])
    loudness = int(spec_params["LOUDNESS"])

    jordan_dataset = int(params["JORDAN_DATASET"])
    pneumonia_only = int(params["PNEUMONIA_ONLY"])

    wav_add = bool(wav_params['ADD'])
    wav_quantity = int(wav_params['QUANTITY'])
    wav_path = str(wav_params["WAV_PATH"]).split(',')

    print("All variables have been collected.")

    train_test_ratio = 0.8

    filenames = []
    labels = []

    wandb_name = __file__.split('/')[-1].split('.')[0] + str('_id{}'.format(job_count))
    # print(wandb_name)
    run = wandb.init(project="tensorboard-integration", name=wandb_name, sync_tensorboard=False)

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
    # augmentation_path = dataset_path.split('/')

    filenames, labels = process_icbhi(six, dataset_path)

    # path = '../../data/txt_datasets/{}'.format(train_file_name)

    nb_pneumonia = labels.count(1)
    nb_non_pneumonia = labels.count(0)
    print("Pneumonia patients: {}, Non-pneumonia patients: {}".format(nb_pneumonia, nb_non_pneumonia))

    # new dataset
    new_dataset_path = '../../data/txt_datasets/all_sw_coch_preprocessed_v2_param_v44_augm_v0_cleaned_8000/'
    _list = os.path.join(new_dataset_path, 'aa_paths_and_labels.txt')

    if jordan_dataset:
        new_filenames = []
        new_labels = []
        with open(_list) as infile:
            for line in infile:
                elements = line.rstrip("\n").rsplit(',', 1)
                filename = elements[0]
                label = float(elements[1])
                if pneumonia_only:
                    if label == 0:
                        continue
                new_filenames.append(filename)
                new_labels.append(label)
        
        jordan_samples = list(zip(new_filenames, new_labels))
        jordan_samples = sorted(jordan_samples)
        jordan_train_samples = jordan_samples[:(int(0.8*len(jordan_samples))+2)]
        jordan_val_samples = jordan_samples[(int(0.8*len(jordan_samples))+2):]
    
    nb_pneumonia = new_labels.count(1)
    nb_non_pneumonia = new_labels.count(0)
    print("Pneumonia patients: {}, Non-pneumonia patients: {}".format(nb_pneumonia, nb_non_pneumonia))

    samples = list(zip(filenames, labels))

    samples = sorted(samples)

    train_samples = samples[:int(0.8*len(samples))]
    val_samples = samples[int(0.8*len(samples)):]

    if jordan_dataset:
        train_samples += jordan_train_samples
        val_samples += jordan_val_samples

    random.shuffle(val_samples)
    random.shuffle(train_samples)

    if testing:
        train_samples = train_samples[:50]
        val_samples = val_samples[:50]
        lr = 1e-5
        n_epochs = 4
        spec_quantity = 10
        wav_quantity = 10

    indices_to_vis = random.sample(range(len(val_samples)), 45)
    samples_to_vis = [val_samples[i] for i in indices_to_vis]

    if concat:
        f = concatenate_specs
        val_dataset, train_dataset = process_data(grouped_val_samples, grouped_train_samples, f, shape, batch_size, initial_channels, cuberooting)
    else:
        print("-----------------------")
        train_samples = [list(sample) for sample in train_samples]
        original_length = len(train_samples)
        train_labels = [label for _, label in train_samples] 
        nb_train_pneumonia, nb_train_non_pneumonia = train_labels.count(1), train_labels.count(0)
        print("Number of train recordings: {} with pneumonia: {} and non-pneumonia: {}".format(original_length, nb_train_pneumonia, nb_train_non_pneumonia))

    
        if augmentation: 
            pneumonia_augmix_samples = apply_augmix_augmentation(train_samples, pneumonia_augmix_quantity, dataset_path, job_version, wav_path, 1, shape)
            non_pneumonia_augmix_samples = apply_augmix_augmentation(train_samples, non_pneumonia_augmix_quantity, dataset_path, job_version, wav_path, 0, shape)
            train_samples.extend(pneumonia_augmix_samples)
            train_samples.extend(non_pneumonia_augmix_samples)

        print("Length of training samples after augmix augmentation: {}".format(len(train_samples)))

        val_labels = [label for _, label in val_samples] 
        nb_val_pneumonia, nb_val_non_pneumonia = val_labels.count(1), val_labels.count(0)
        print("Number of val recordings: {} with pneumonia: {} and non-pneumonia: {}".format(len(val_samples), nb_val_pneumonia, nb_val_non_pneumonia))
        print("-----------------------")

        f = custom_generate_spec
        # val_dataset, train_dataset = process_data(val_samples, train_samples, f, shape, batch_size, initial_channels, cuberooting, normalizing=normalizing)
        train_filenames, train_labels = zip(*train_samples)
        val_filenames, val_labels = zip(*val_samples)

        train_dataset = tf.data.Dataset.from_tensor_slices((list(train_filenames), list(train_labels)))
        train_dataset = train_dataset.shuffle(len(train_filenames))
        train_dataset = train_dataset.map(lambda filename, label: generate_spec(filename, label, shape, initial_channels, cuberooting, normalizing), num_parallel_calls=4)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(1)

        val_dataset = tf.data.Dataset.from_tensor_slices((list(val_filenames), list(val_labels)))
        val_dataset = val_dataset.shuffle(len(val_filenames))
        val_dataset = val_dataset.map(lambda filename, label: custom_generate_spec(filename, label, shape, initial_channels, cuberooting, normalizing), num_parallel_calls=4)
        val_dataset = val_dataset.batch(1)
        val_dataset = val_dataset.prefetch(1)

        train_dataset = train_dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))

        mixup_quantity = 158*repeat_factor
        alpha = 0.4
        both_classes = True

        if both_classes:
            pneumonia_train_dataset = train_dataset.filter(lambda x, y: y == 1)
            non_pneumonia_train_dataset = train_dataset.filter(lambda x, y: y == 0)
            train_slice_one = pneumonia_train_dataset.take(158)
            train_slice_one = train_slice_one.repeat(repeat_factor) 
            train_slice_two = non_pneumonia_train_dataset.take(mixup_quantity)
        else:
            train_slice_one = train_dataset.take(mixup_quantity) 
            train_slice_two = train_dataset.take(mixup_quantity)
        
        # train_slice_one = train_slice_one.shuffle(mixup_quantity)
        # train_slice_two = train_slice_two.shuffle(mixup_quantity)

        train_slice = tf.data.Dataset.zip((train_slice_one, train_slice_two))

        mixup_batch_size = 1

        mixup_train_dataset = train_slice.map(lambda ds_one, ds_two: custom_mix_up(ds_one, ds_two, batch_size=mixup_batch_size, alpha=alpha), num_parallel_calls=tf.data.AUTOTUNE)

        train_dataset = train_dataset.concatenate(mixup_train_dataset)
        
        mixup_length = len(list(train_dataset.as_numpy_iterator()))
        print("New dataset length after mix-up augmentation: {}".format(mixup_length))

        train_dataset = train_dataset.shuffle(mixup_length)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(1)

    # weights
    weights = None

    if bool(params["CLASS_WEIGHTS"]):
        print("Initializing weights...")
        weights = class_weight.compute_class_weight(
            "balanced", np.unique(labels), labels)
        weights = {i: weights[i] for i in range(0, len(weights))}
        print("weights = {}".format(weights))
    
    # callbacks
    metrics_callback = custom_metric_callback(val_dataset, shape, initial_channels, n_classes, job_dir, sr, min_delta, es_patience, patience, min_lr, factor, epoch_start, target, job_count, clause, samples_to_vis, f, cuberooting, normalizing, adaptive_lr=adaptive_lr)

    gpus = tf.config.experimental.list_logical_devices('GPU')

    # model setting
    model = model_to_be_trained(**model_params)

    model.summary(line_length=110)

    if len(gpus) > 1:
        print("You are using 2 GPUs while the code is set up for one only.")
        exit()
    model.fit(
        train_dataset,
        epochs=n_epochs,
        verbose=2,
        class_weight=weights,
        callbacks=[metrics_callback]
    )

    model.save(job_dir + "/{}/model_{}.h5".format(job_count, n_epochs))

    run.finish()

if __name__ == "__main__":
    print("Tensorflow Version: {}".format(tf.__version__))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
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
    train_file = arguments.pop("train_file")
    job_dir = arguments.pop("job_dir")
    params = arguments.pop("params")
    params = json.loads(params)
    wav_params = arguments.pop("wav_params")
    wav_params = json.loads(wav_params)
    spec_params = arguments.pop("spec_params")
    spec_params = json.loads(spec_params)
    print("-----------------------")
    print("Using module: {}".format(__file__[-15:]))
    print("Using train_file: {}".format(train_file))
    print("Job directory: {}".format(job_dir))
    print("Parameters: {}".format(params))
    print("Augmentation parameters for audio: {}".format(wav_params))
    print("Augmentation parameters for spectrograms: {}".format(spec_params))
    print("-----------------------")
    train_model(train_file, job_dir, params, wav_params, spec_params, mixednet, 0, 158, 400, repeat_factor=4)
    print("-----------------------")
    print("158/800 with v42")
    wav_params['WAV_PATH'] = 'v42'
    print(wav_params)
    train_model(train_file, job_dir, params, wav_params, spec_params, mixednet, 1, 158, 400, repeat_factor=4)
    print("-----------------------")
    print("158/800 with v42,v43")
    wav_params['WAV_PATH'] = 'v42,v43'
    print(wav_params)
    train_model(train_file, job_dir, params, wav_params, spec_params, mixednet, 2, 158, 400, repeat_factor=4)

