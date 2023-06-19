import sys
# sys.path.insert(0, 'main/models/conv')
from modules.main import parameters
from modules.main.helpers import *
from modules.main.global_helpers import *

from modules.audio_loader.JordanAudioLoader import JordanAudioLoader
from modules.audio_loader.IcbhiAudioLoader import IcbhiAudioLoader
from modules.audio_loader.BdAudioLoader import BdAudioLoader
from modules.audio_loader.PerchAudioLoader import PerchAudioLoader
from modules.audio_loader.helpers import default_get_filenames, bd_get_filenames, perch_get_filenames

from modules.spec_generator.SpecGenerator import SpecGenerator

# from modules.callbacks.NewCallback import NewCallback

from modules.models.leaf_model9_model import leaf_model9_model
from modules.models.leaf_mixednet_model import leaf_mixednet_model

from modules.parse_functions.parse_functions import spec_parser, generate_timed_spec

from modules.augmenter.functions import stretch_time, shift_pitch 
from modules.models.core import Distiller

from collections import Counter
from sklearn.utils import class_weight
import tensorflow as tf
import tensorflow_addons as tfa
import leaf_audio.frontend as frontend
import numpy as np

from matplotlib import pyplot as plt
import numpy as np
import shutil

import os
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.keras import backend as K
from tensorflow.keras.utils import to_categorical
import types
from collections import defaultdict
from tabulate import tabulate
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, precision_recall_curve, auc, plot_precision_recall_curve, PrecisionRecallDisplay, roc_curve, plot_roc_curve, RocCurveDisplay
import itertools
import librosa
from librosa import display
import soundfile as sf
# import wandb
import matplotlib
import matplotlib.cm as plt_cm
matplotlib.use('Agg')
import PIL
from ..spec_generator.SpecGenerator import SpecGenerator

np.set_printoptions(precision=3)

class NewCallback(tf.keras.callbacks.LambdaCallback):
    '''
    Callback that handles all the validation processes, such as managing adaptive lr and early stopping, generating metrics, and creating patient reports
    '''

    def __init__(self, validation_data, val_filenames, target_key="avg_accuracy", average_mode='binary', clause_portion=0.9):

        self.validation_data = validation_data
        self.val_filenames = val_filenames
        self.visualize_spec = types.MethodType(visualize_spec, self)
        self.average_mode = average_mode
        self.clause_portion = clause_portion
        self.target_key = target_key
        self.best_metrics_dict = defaultdict(lambda: 0)
        self.job_dir = None 
        self.best_model = None
        self.tracker = 0
        self.tracker_metric = 0
    
    def on_train_begin(self, logs=None):
        "job id is {}".format(parameters.job_id)
        # self.job_dir = os.path.join(parameters.file_dir, str(parameters.job_id))
        self.job_dir = parameters.job_dir
        print("Target metric is {}".format(self.target_key))

    def on_train_end(self, logs=None):
        tf.print("----------------------------------")
        tf.print("The best metrics for this job are: \n")
        for metric_name, metric_value in self.best_metrics_dict.items():
            print(metric_name)
            if metric_name == "model":
                continue
            tf.print("Best {}: {}".format(metric_name, metric_value))
        # wandb.init(project="tensorboard-integration", sync_tensorboard=False)
        # wandb.save(self.job_dir + "/report.txt")

    def on_epoch_end(self, epoch, logs=None):

        one_indexed_epoch = epoch + 1

        if parameters.mode == "cw":
            y_true_all, preds_all, rounded_preds_all, __, patients_dict = self.return_key_cw_elements(one_indexed_epoch)
        else:
            y_true_all, preds_all, rounded_preds_all, __, patients_dict = self.return_key_elements(one_indexed_epoch)

        cm, normalized_cm, cm_sum, acc, class_accuracies = self.return_base_metrics(y_true_all, rounded_preds_all)

        if not (parameters.mode == "cw"):
            precision, recall, f1, __ = self.return_precision_recall_f1(y_true_all, rounded_preds_all)
            roc_fpr, roc_tpr, roc_auc = self.return_roc_metrics(y_true_all, preds_all)
        # self.make_roc_curve(x_all, y_true_all, roc_fpr, roc_tpr)

        avg_accuracy = np.sum(class_accuracies) / len(class_accuracies)

        if parameters.mode == "cw":
            metrics_dict = self.make_metrics_dict(cm, normalized_cm, acc, class_accuracies, avg_accuracy, one_indexed_epoch)
        else:
            metrics_dict = self.make_metrics_dict(cm, normalized_cm, acc, class_accuracies, avg_accuracy, one_indexed_epoch, precision, recall, f1, roc_auc,)

        for metric_name, metric_value in metrics_dict.items():
            tf.print("Validation {}: {}".format(metric_name, metric_value))

        target_metric, best_target_metric = self.select_target_metric(metrics_dict)

        if self.apply_clause(cm, cm_sum) == 1:
            return 

        if one_indexed_epoch >= parameters.epoch_start: 
            self.early_stopping_info(target_metric)
            if parameters.adaptive_lr:
                self.adaptive_lr_info()
        if ((one_indexed_epoch >= parameters.epoch_start) and (target_metric > best_target_metric)) or parameters.testing:
            self.update_best_results(metrics_dict, patients_dict, one_indexed_epoch)
    
    def update_best_results(self, metrics_dict, patients_dict, one_indexed_epoch):
        print("-- New best results were achieved. --")
        self.best_metrics_dict["model"] = self.model
        for key, value in metrics_dict.items():
            self.best_metrics_dict[key] = value
        # write tns, fns, etc AND grad cam visualizations
        # self.write_cases(patients_dict) #TODO: DECOMMENT
        # patient report for pneumonia only thus far
        if not (parameters.mode == "cw"):
            best_patient_cm = self.generate_patient_report(patients_dict)
            self.best_metrics_dict["patient_cm"] = best_patient_cm
        print("Saving model...")
        # self.model.save(self.job_dir + "/model_epoch{}.h5".format(one_indexed_epoch)) #TODO: DECOMMENT

    def make_metrics_dict(self, cm, normalized_cm, acc, class_accuracies, avg_accuracy, one_indexed_epoch, precision=None, recall=None, f1=None, roc_auc=None ):
    # def make_metrics_dict(self, **kwargs):
        return {"cm": cm, "normalized_cm": normalized_cm, "acc": acc, "class_accuracies": class_accuracies, 
            "precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc, "avg_accuracy": avg_accuracy, "one_indexed_epoch": one_indexed_epoch}

    def select_target_metric(self, metrics_dict):

        target_metric = metrics_dict[self.target_key]
        best_target_metric = self.best_metrics_dict[self.target_key]

        return target_metric, best_target_metric

    def make_roc_curve(self, x_all, y_true_all, roc_fpr, roc_tpr,):
        roc_display = plot_roc_curve(self.model, x_all, y_true_all)
        plt.xticks(np.arange(0, 1, 0.1))
        plt.yticks(np.arange(0, 1, 0.1))
        plt.close()
        RocCurveDisplay.plot()
        pr_display = RocCurveDisplay(fpr, tpr, roc_auc=roc_auc)
        pr_display = pr_display.plot()
        pr_display.ax_.get_legend().remove()
        pr_display.ax_.xaxis.set_ticks(np.arange(0, 1, 0.1))
        pr_display.ax_.yaxis.set_ticks(np.arange(0, 1, 0.1))
        path = os.path.join(self.job_dir, "test_pr_curve_{}.png".format(one_indexed_epoch))
        path = "pr_curves/test_pr_curve_{}.png".format(one_indexed_epoch)

    def return_roc_metrics(self, y_true_all, preds_all):
        roc_fpr, roc_tpr, roc_thresholds = roc_curve(y_true_all, preds_all)
        roc_auc = auc(roc_fpr, roc_tpr)
        return roc_fpr, roc_tpr, roc_auc
    
    def return_precision_recall_f1(self, y_true_all, rounded_preds_all):
        return precision_recall_fscore_support(y_true_all, rounded_preds_all, average=self.average_mode)

    def convert(self, el):
        if el[0] == 0 and el[1] == 0:
            return 0
        elif el[0] == 1 and el[1] == 0:
            return 1
        elif el[0] == 0 and el[1] == 1:
            return 2
        elif el[0] == 1 and el[1] == 1:
            return 3
        
    def return_base_metrics(self, y_true_all, rounded_preds_all):
        cm = np.zeros((parameters.n_classes + 1, parameters.n_classes + 1))
        if parameters.mode == "cw":
            print(rounded_preds_all[0])
            y_true_all = [self.convert(y_true[0]) for y_true in y_true_all]
            rounded_preds_all = [self.convert(rounded_pred[0]) for rounded_pred in rounded_preds_all]
        cm = confusion_matrix(y_true_all, rounded_preds_all)
        normalized_cm = confusion_matrix(y_true_all, rounded_preds_all, normalize='true')
        cm_sum = np.sum(cm)
        acc = np.trace(cm) / cm_sum
        class_accuracies=self.compute_class_accuracies(cm)
        return cm, normalized_cm, cm_sum, acc, class_accuracies

    def write_cases(self, patients_dict):

        if (not ((self.model.name == "time_series_model"))): # or (self.model.name == "audio_model"))):
            #### initializing the folders propery
            if os.path.exists(os.path.join(self.job_dir, "fp")):
                    shutil.rmtree(os.path.join(self.job_dir, "fp"))
                    shutil.rmtree(os.path.join(self.job_dir, "fn"))
                    shutil.rmtree(os.path.join(self.job_dir, "tp"))
                    shutil.rmtree(os.path.join(self.job_dir, "cw"))
            os.mkdir(os.path.join(self.job_dir, "fp"))
            os.mkdir(os.path.join(self.job_dir, "fn"))
            os.mkdir(os.path.join(self.job_dir, "tp"))
            os.mkdir(os.path.join(self.job_dir, "cw"))
        
            for patient_id, patient_recordings in patients_dict.items():
                for i, r in enumerate(patient_recordings):

                    #### gradcam visualization
                    # if i % 10 == 0:
                    #     expanded_spec = np.expand_dims(r[0], axis=(0, -1))
                    #     create_gradcam_heatmap(expanded_spec, self.model, os.path.join(self.job_dir, "gradcam/", r[1])) #TODO: DECOMMENT

                    if r[4] == "tn" or len(r) == 0:
                        continue

                    #### saving tps, fps, and fns spectrograms to respective folders
                    dest = os.path.join(self.job_dir, r[4], r[1])
                    self.visualize_spec(r[0], parameters.sr, dest, title = str(r[2]) + "__" + str(r[3]))
        
    def return_patient_id(self, filename):
        if filename.startswith("0"):
                return filename[:8]
        else:
            return filename.split('_')[0]

    def return_spectrogram(self, audio_c, layer_class_name):

        layer_name = next((layer.name for layer in self.model.layers if layer.__class__.__name__ == layer_class_name), None) # if layer.__class__.__name__ == layer_class_name
        partial_model = tf.keras.Model(self.model.input, self.model.get_layer(layer_name).output)
        spec = partial_model([audio_c], training=False)
        spec = np.squeeze(spec.numpy())
        return spec

    def return_key_elements(self, one_indexed_epoch):

        y_true_all = []
        preds_all = []
        rounded_preds_all = []
        x_all = []
        patients_dict = defaultdict(list) # struct is { patient_id: [[name, y_pred, y_true, "tn" or "fn" or "fp" or "tp"]] }
        
        for i, (x, y_true) in enumerate(self.validation_data):
            x = x.numpy()
            x_copy = x.copy()
            y_true = y_true.numpy()
            filename = self.val_filenames[i]
            patient_id = self.return_patient_id(filename)
            pred = self.model.predict(x)
            # TODO: MAKE THIS A DIFFERENT FUNCTION SOMEWHERE ELSE WHEN REWORKING ON EXTRACTING SPECS AND FUNCTIONS TO ANOTHER FOLDER
            if self.model.name == "audio_model":
                # print("LIBROSA")
                # test = SpecGenerator([], "random")
                # spec = test.generate_mel_spec(np.squeeze(x))
                # print(spec.shape)
                # print(np.sum(spec))
                # self.visualize_spec(spec, parameters.sr, os.path.join(self.job_dir, "patient_avg_specs/"), title = "orig")
                # print("KAPRE")
                if 'ApplyFilterbank' in [layer.__class__.__name__ for layer in self.model.layers]:
                    x = self.return_spectrogram(x, 'ApplyFilterbank')
                    x = np.transpose(x)
                    x = librosa.power_to_db(x, ref=np.max)
                ##### Visualization
                if i % 15 == 0:
                    if 'ApplyFilterbank' in [layer.__class__.__name__ for layer in self.model.layers]:
                        self.visualize_spec(x, parameters.sr, os.path.join(self.job_dir, "others/"), title = "{}_ind{}_epo{}".format(filename, i, one_indexed_epoch ))
                    if 'MagnitudeToDecibel' in [layer.__class__.__name__ for layer in self.model.layers]:
                        x_new = self.return_spectrogram(x_copy, 'MagnitudeToDecibel')
                        x_new = np.transpose(x_new)
                        self.visualize_spec(x_new, parameters.sr, os.path.join(self.job_dir, "others/"), title = "{}_decibel_ind{}_epo{}".format(filename, i, one_indexed_epoch ))

            pred = pred.reshape(len(pred))
            rounded_pred = np.zeros(pred.shape)
            rounded_pred[pred >= 0.5] = 1
            rounded_pred[pred < 0.5] = 0
            x = np.squeeze(x)
            if y_true == 0 and rounded_pred == 0: # tn
                patients_dict[patient_id].append([x, filename, pred, y_true, "tn"])
            if y_true == 1 and rounded_pred == 0: # fn
                patients_dict[patient_id].append([x, filename, pred, y_true, "fn"])
            if y_true == 0 and rounded_pred == 1: # fp
                patients_dict[patient_id].append([x, filename, pred, y_true, "fp"])
            if y_true == 1 and rounded_pred == 1: # tp
                patients_dict[patient_id].append([x, filename, pred, y_true, "tp"])
            y_true_all.append(y_true)
            rounded_preds_all.append(rounded_pred)
            preds_all.extend(pred)
            x_all.append(x)
        
        return y_true_all, preds_all, rounded_preds_all, x_all, patients_dict
    
    def adaptive_lr_info(self):  
        # print(self.model.layer[0].optimizer)
        # print(self.model.layer[1].optimizer)
        # print(self.model.optimizer[0])
        # print(self.model.optimizer[1])
        # exit()
        print(self.model.optimizer)
        exit()
        if (not (self.tracker == 0)) and self.tracker % parameters.lr_patience == 0:
            print(self.)
                if self.model.optimizer.lr > parameters.min_lr:
                    self.model.optimizer.lr = self.model.optimizer.lr * parameters.factor
                    print("Lr has been adjusted to {}".format(self.model.optimizer.lr.numpy()))
    
    def apply_clause(self, cm, cm_sum):
        column_sums = np.sum(cm, axis=0)
        if parameters.clause:
            print("The clause is activated.")
            if (column_sums[0] >= (self.clause_portion * cm_sum)) or (column_sums[1] >= (self.clause_portion * cm_sum)) or (column_sums[1] >= (0.6 * cm_sum)):
                print("The training is defaulting to either class.")
                return 1
            else:
                print("Not defaulting.")
                return 0
    
    def early_stopping_info(self, current_metric):
        diff = current_metric - self.tracker_metric
        if diff > parameters.min_delta:
            self.tracker = 0
            self.tracker_metric = current_metric
        else:
            self.tracker += 1
            if self.tracker == parameters.es_patience:
                print("The number of epochs since last 1% equals the patience")
                self.model.stop_training = True
            else:
                print("The validation tracker metric at {} hasn't increased by {} in {} epochs".format(self.tracker_metric, parameters.min_delta, self.tracker))
        return None

    def generate_patient_report(self, patients_dict, names=["Patient id", "status", "tns", "fns", "fps", "tps", "number of recordings", "average score"]):
        
        tn_path = self.job_dir + "/tns.txt"
        report_path = self.job_dir + "/report.txt"
        tn_writer = open(tn_path, "w")
        reports_writer = open(report_path, "w")
        y_true = []
        y_pred = []
        tables_to_write = defaultdict(list)
        # reports_to_write = []
        # tn_to_write = []
        # status_count = defaultdict(lambda: [0,0,0,0])
        for patient_id, patient_recordings in patients_dict.items():
            patient_labels = []
            avg_patient_score = 0
            avg_patient_spec = np.zeros(patient_recordings[0][0].shape)

            #### collecting average scores and spectrograms
            for recording in patient_recordings:
                avg_patient_score += recording[2]
                avg_patient_spec += recording[0]
                patient_labels.append(recording[4])
                # TODO: get average spec & score and save to
            avg_patient_score = avg_patient_score / len(patient_recordings)
            avg_patient_spec = avg_patient_spec / len(patient_recordings)

            #### visualizating average patient spec
            # if (not (self.model.name == "time_series_model")): # or self.model.name == "audio_model")):
            #     self.visualize_spec(avg_patient_spec, parameters.sr, os.path.join(self.job_dir, "patient_avg_specs/"), title = str(patient_id))

            fp = patient_labels.count("fp")
            tp = patient_labels.count("tp")
            fn = patient_labels.count("fn")
            tn = patient_labels.count("tn")
            label = patient_recordings[0][3]
            if fp > 0 or tp > 0:
                pred_class = 1
            else:
                pred_class = 0
            y_true.append(label)
            y_pred.append(pred_class)
            if parameters.mode == "cw":
                status = "n/a"
            else:
                status = self.return_status(label, pred_class)
            # status_count[status][0] += fp
            # status_count[status][1] += tp
            # status_count[status][2] += fn
            # status_count[status][3] += tn
            # tables_to_write[status].append([patient_id, status, str(tn), str(fn), str(fp), str(tp), str(len(patient_recordings)), avg_patient_score])
            tables_to_write[status].append([patient_id, status, tn, fn, fp, tp, len(patient_recordings), avg_patient_score])
            # if status == "tn":
            #     tn_to_write.append([patient_id, status, str(tn), str(fn), str(fp), str(tp), str(len(patient_recordings)), avg_patient_score])
            # else:
            #     reports_to_write.append([patient_id, status, str(tn), str(fn), str(fp), str(tp), str(len(patient_recordings)), avg_patient_score])
        
        # for key, items in status_count.items():
        #     print(key)
        #     print(items)
        # exit()
        for status, table in tables_to_write.items():
            last_row = [0, 0, 0, 0, 0, 0]
            for row in table:
                # print(row)
                last_row = [sum(x) for x in zip(last_row, row[2:])]
                # print(last_row)
            last_row.insert(0, "N/A")
            last_row.insert(0, "Total")
            last_row[-1] = last_row[-1] / len(table)
            table.append(last_row)
            if status == "tn":
                tn_writer.write(tabulate(table, headers=names))
            else:
                reports_writer.write(tabulate(table, headers=names))
                reports_writer.write("\n\n")

            
        # reports_to_write.sort(key = lambda line: line[1])
        # print(reports_to_write)
        # status_reports = np.split(reports_to_write, np.flatnonzero(reports_to_write[:-1][1] != reports_to_write[1:][1])+1)
        # print(a[0])
        # print(a[1])
        # exit()
        # reports_writer.write(tabulate(reports_to_write, headers=names))

        cm = confusion_matrix(y_true, y_pred)
        normalized_cm = confusion_matrix(y_true, y_pred, normalize='true')
        cm_sum = np.sum(cm)
        acc = np.trace(cm) / cm_sum
        class_accuracies=self.compute_class_accuracies(cm)
        message = "Patient Confusion matrix: \n {} ".format(cm)
        print(message)
        reports_writer.write(message)
        message = "Patient Normalized Confusion matrix: \n {} ".format(normalized_cm)
        print(message)
        reports_writer.write(message)
        message = "Patient Validation accuracy: \n {:.2f} ".format(acc*100)
        print(message)
        reports_writer.write(message)
        message = "Patient Validation class accuracies: \n {} ".format(class_accuracies)
        print(message)
        reports_writer.write(message)
        tn_writer.close()
        reports_writer.close()
        return cm
        
    def return_status(self, label, pred_class):
        if label == 0 and pred_class == 0:
            return "tn"
        if label == 0 and pred_class == 1:
            return "fp"
        if label == 1 and pred_class == 0:
            return "fn"
        if label == 1 and pred_class == 1:
            return "tp"
            

    def compute_class_accuracies(self, cm):
        accs = []
        for i, row in enumerate(cm):
            accs.append(row[i] / sum(row))
        return accs
    
    def return_key_cw_elements(self, one_indexed_epoch):

        y_true_all = []
        preds_all = []
        rounded_preds_all = []
        x_all = []
        patients_dict = defaultdict(list) # struct is { patient_id: [[name, y_pred, y_true, "tn" or "fn" or "fp" or "tp"]] }
        
        for i, (x, y_true) in enumerate(self.validation_data):
            x = x.numpy()
            x_copy = x.copy()
            y_true = y_true.numpy()
            filename = self.val_filenames[i]
            patient_id = self.return_patient_id(filename)
            pred = self.model.predict(x)
            if self.model.name == "audio_model":
                # print("LIBROSA")
                # test = SpecGenerator([], "random")
                # spec = test.generate_mel_spec(np.squeeze(x))
                # print(spec.shape)
                # print(np.sum(spec))
                self.visualize_spec(spec, parameters.sr, os.path.join(self.job_dir, "patient_avg_specs/"), title = "orig")
                # print("KAPRE")
                x = self.return_spectrogram(x, 'ApplyFilterbank')
                x = np.transpose(x)
                # x = librosa.power_to_db(x, ref=np.max)
                if i % 50 == 0:
                    self.visualize_spec(x, parameters.sr, os.path.join(self.job_dir, "others/"), title = "{}_ind{}_epo{}".format(filename, i, one_indexed_epoch ))
                    if 'MagnitudeToDecibel' in [layer.__class__.__name__ for layer in self.model.layers]:
                        x_new = self.return_spectrogram(x_copy, 'MagnitudeToDecibel')
                        x_new = np.transpose(x_new)
                        self.visualize_spec(x_new, parameters.sr, os.path.join(self.job_dir, "others/"), title = "{}_decibel_ind{}_epo{}".format(filename, i, one_indexed_epoch ))
                # print(x.shape)
                # print(np.sum(x))
                # exit()
            # pred = pred.reshape(len(pred))
            rounded_pred = np.zeros(pred.shape)
            rounded_pred[pred >= 0.5] = 1
            rounded_pred[pred < 0.5] = 0
            x = np.squeeze(x)
            # if y_true == (0,0) and rounded_pred == (0, 0): # tn
            #     patients_dict[patient_id].append([x, filename, pred, y_true, "tn"])
            # if y_true == (0, 1) and rounded_pred == (0, 0): # fn
            #     patients_dict[patient_id].append([x, filename, pred, y_true, "fn"])
            # if y_true == 0 and rounded_pred == 1: # fp
            #     patients_dict[patient_id].append([x, filename, pred, y_true, "fp"])
            # if y_true == 1 and rounded_pred == 1: # tp
            #     patients_dict[patient_id].append([x, filename, pred, y_true, "tp"])
            # print(filename)
            # print(pred)
            # print(rounded_pred)
            # print(y_true)
            patients_dict[patient_id].append([x, filename, pred, y_true, "cw"])
            y_true_all.append(y_true)
            rounded_preds_all.append(rounded_pred)
            preds_all.extend(pred)
            x_all.append(x)
        
        return y_true_all, preds_all, rounded_preds_all, x_all, patients_dict

def return_layer(model):

    if model.name == "mixednet":
        return model.layers[1].name
    elif model.name == "model9":
        return "inverted_residual_6"
    elif model.name == "time_series_model":
        return "time_distributed_3"
    elif model.name == "audio_model":
        return "apply_filterbank"

def create_gradcam_heatmap(spec, model, dest, pred_index=None):

    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    layer_name = return_layer(model)
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(spec)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    axes = (0, 1, 2, 3) if (model.name == "time_series_model") else (0, 1, 2)
    pooled_grads = tf.reduce_mean(grads, axis=axes)

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    heatmap = heatmap.numpy()
    
    save_and_display_gradcam(spec, model, heatmap, dest)

def save_and_display_gradcam(spec, model, heatmap, dest, alpha=0.4):
    # Load the original image
    # spec = keras.preprocessing.image.load_spec(spec_path)
    # spec = keras.preprocessing.image.spec_to_array(spec)

    # Rescale heatmap to a range 0-255
    # print("inside save")
    # print(heatmap.shape)

    spec = np.squeeze(spec, axis=0)
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = plt_cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    # print(jet_heatmap.shape)
    
    # if model.name == "time_series_model":
    #     print(spec.shape)
    #     new_spec_axis = tf.shape(spec)[0] * tf.shape(spec)[1]
    #     spec=  tf.reshape(spec, shape=(new_spec_axis, tf.shape(spec)[2], -1))
    #     print(spec.shape)
    #     new_axis = tf.shape(jet_heatmap)[0] * tf.shape(jet_heatmap)[1]
    #     jet_heatmap = tf.reshape(jet_heatmap, shape=(new_axis, tf.shape(jet_heatmap)[2], -1))
    # print(jet_heatmap.shape)
    
    if model.name == "time_series_model":
        for i in range(tf.shape(spec)[0]):
            ts_jet_heatmap = jet_heatmap[i, :, :]
            ts_spec = spec[i, :, :]
            ts_jet_heatmap = tf.keras.preprocessing.image.array_to_img(ts_jet_heatmap)
            # print(ts_jet_heatmap.size)
            ts_jet_heatmap = ts_jet_heatmap.resize((ts_spec.shape[1], ts_spec.shape[0]))
            # print(ts_jet_heatmap.size)
            ts_jet_heatmap = tf.keras.preprocessing.image.img_to_array(ts_jet_heatmap)

            # Superimpose the heatmap on original image
            superimposed_ts_spec = ts_jet_heatmap * alpha + ts_spec
            superimposed_ts_spec = tf.keras.preprocessing.image.array_to_img(superimposed_ts_spec)

            # flip it
            superimposed_ts_spec = superimposed_ts_spec.transpose(PIL.Image.FLIP_TOP_BOTTOM)

            superimposed_ts_spec.save(dest + "_{}.png".format(i))
    # Create an image with RGB colorized heatmap
    else:
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        # print(jet_heatmap.size)
        jet_heatmap = jet_heatmap.resize((spec.shape[1], spec.shape[0]))
        # print(jet_heatmap.size)
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_spec = jet_heatmap * alpha + spec
        superimposed_spec = tf.keras.preprocessing.image.array_to_img(superimposed_spec)

        # flip it
        superimposed_spec = superimposed_spec.transpose(PIL.Image.FLIP_TOP_BOTTOM)

        # Save the superimposed image

        superimposed_spec.save(dest + ".png")

    # Display Grad CAM
    # display(Image(cam_path))

def train_model(datasets, model_to_be_trained, spec_aug_params, audio_aug_params, parse_function):

    
    # simply initialize audio loader object for each dataset
    # mandatory parameters:  (1) root of dataset (2) function for extracting filenames 
    # optional parameters: or other custom parameters, like the Bangladesh excel path
    # NOTE: name attribute: to distinguish between datasets when the same audio loader object is used for different datasets, such as antwerp and icbhi that both use IcbhiAudioLoader

    audio_loaders = []
    
    if datasets["Jordan"]: audio_loaders.append(JordanAudioLoader(parameters.jordan_root, default_get_filenames))
    if datasets["Bd"]: audio_loaders.append(BdAudioLoader(parameters.bd_root, bd_get_filenames, parameters.excel_path))
    if datasets["Perch"]: audio_loaders.append(PerchAudioLoader(parameters.perch_root, perch_get_filenames))
    if datasets["Icbhi"]: audio_loaders.append(IcbhiAudioLoader(parameters.icbhi_root, default_get_filenames))
    if datasets["Ant"]: 
        # TODO: pass names?
        ant_loader = IcbhiAudioLoader(parameters.ant_root, default_get_filenames)
        ant_loader.name = "Antwerp"
        audio_loaders.append(ant_loader)
        
    # this functions loads the audios files from the given input, often .wav and .txt files
    # input: [filename1, filename2, ...]
    # output: {'Icbhi': [[audio1, label2, filename1], [audio2, label2, filename2], 'Jordan:' : ... }
    audios_dict = load_audios(audio_loaders)
    # print(audios_dict)
    
    # ths function takes the full audios and prepares its N chunks accordingly
    # by default, it returns samples grouped by patient according to the respective logics of datasets
    # input: [[audio1, label1, filename1], [audio2, label2, filename2], ...]
    # output: [ [all chunks = [audio, label, filename] of all files for patient1], [same for patient 2], ...]
    audios_c_dict = prepare_audios(audios_dict)
    # print(audios_c_dict)

    # NOTE: # Data is grouped by dataset and patient thus far
    # this functions (1) splits each dataset into train and validation, then (2) after split, we don't care about grouping by patient = flatten to list of audios by patients to give a list of audios 
    #  input: Full Dictionary:  {Icbhi: [] -> data grouped by PATIENT, Jordan: [] -> data grouped by PATIENT, ...}
    # output: Training /// Val  dictionary:   {Icbhi: [] -> data organized INDIVIDUALLY, Jordan: [] -> data organized  INDIVIDUALLY} 
    train_audios_c_dict, val_audios_c_dict = split_and_extend(audios_c_dict, parameters.train_test_ratio)
    # NOTE: # Data is only grouped by dataset now
    

    # simplest step: now that everything is ready, we convert to spectrograms! it's the most straightforward step...
    # convert: [audio, label, filename] -> [SPEC, label, filename]
    val_samples = generate_audio_samples(val_audios_c_dict)
    # ... but it's different for training because of augmentation. the following function sets up and merges 2 branches:
    #   1) augment AUDIO and convert to spectrogram
    #   2) convert to spectrogram and augment SPECTROGRAM
    # train_samples, original_training_length = set_up_training_samples(train_audios_c_dict, spec_aug_params, audio_aug_params) 
    train_samples = generate_audio_samples(train_audios_c_dict)
    # train_samples = generate_spec_samples(train_audios_c_dict) # the same as above if no augmentation 
     # NOTE: # Data is NOT LONGER grouped by dataset 

    # from now on it's cake!
    train_dataset, __, train_labels, __ = create_tf_dataset(train_samples, batch_size=parameters.batch_size, shuffle=True, parse_func=parse_function)
    val_dataset, val_specs, val_labels, val_filenames = create_tf_dataset(val_samples, batch_size=1, shuffle=False, parse_func=parse_function) # keep shuffle = False!
    train_non_pneumonia_nb, train_pneumonia_nb = train_labels.count(0), train_labels.count(1)
    # print(list(train_dataset.as_numpy_iterator())[0])
    # print(list(train_dataset.as_numpy_iterator())[0][0].shape)
    # exit()
    print("-----------------------")
    print_dataset(train_labels, val_labels)

    # weights
    weights = None
    
    # handles metrics, file saving (all the files inside gradcam/, tp/, others/, etc), report writing (report.txt), visualizations, etc
    parameters.adaptive_lr = False
    metrics_callback = NewCallback(val_dataset, val_filenames)

    gpus = tf.config.experimental.list_logical_devices('GPU')

    # teacher
    teacher = leaf_model9_model(num_outputs=parameters.n_classes, frontend=frontend.Leaf(sample_rate=16000, n_filters=80), encoder=None)
    shape = (None, parameters.audio_length*parameters.sr)
    teacher.build(shape)

    optimizers = [
        tf.keras.optimizers.Adam(
            lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=parameters.epsilon, decay=parameters.weight_decay, amsgrad=False
        ),
        tf.keras.optimizers.Adam(
            lr=parameters.lr, beta_1=0.9, beta_2=0.999, epsilon=parameters.epsilon, decay=parameters.weight_decay, amsgrad=False
        )
    ]
    optimizers_and_layers = [(optimizers[0], teacher.layers[0]), (optimizers[1], teacher.layers[1:])]
    opt = tfa.optimizers.MultiOptimizer(optimizers_and_layers)

    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=parameters.label_smoothing)

    teacher.compile(
        optimizer=opt,
        loss=loss,
        metrics=[
            'accuracy', 
        ],
    )

    teacher.summary(line_length=110)

    teacher.fit(
        train_dataset,
        epochs=parameters.n_epochs,
        verbose=2,
        class_weight=weights,
        callbacks=[metrics_callback]
    )

    #student
    student = leaf_mixednet_model(num_outputs=parameters.n_classes, frontend=frontend.Leaf(sample_rate=16000, n_filters=80), encoder=None)
    shape = (None, parameters.audio_length*parameters.sr)
    student.build(shape)

    optimizers = [
        tf.keras.optimizers.Adam(
            lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=parameters.epsilon, decay=parameters.weight_decay, amsgrad=False
        ),
        tf.keras.optimizers.Adam(
            lr=parameters.lr, beta_1=0.9, beta_2=0.999, epsilon=parameters.epsilon, decay=parameters.weight_decay, amsgrad=False
        )
    ]
    optimizers_and_layers = [(optimizers[0], student.layers[0]), (optimizers[1], student.layers[1:])]
    opt = tfa.optimizers.MultiOptimizer(optimizers_and_layers)

    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=parameters.label_smoothing)

    student.compile(
        optimizer=opt,
        loss=loss,
        metrics=[
            'accuracy', 
        ],
    )

    student.summary(line_length=110)


    folder = "{}/others".format(parameters.job_dir)
    sps = []
    s = list(train_dataset.as_numpy_iterator())[0][0][:3]
    for i, sp in enumerate(s):
        sp = np.expand_dims(sp, axis=0)
        sp = student(sp, return_spec=True)
        sp = np.swapaxes(np.squeeze(sp.numpy()), 0, 1)
        sps.append(sp)
        visualize_spec_bis(sp, sr=parameters.sr, dest="{}/backend_before_{}".format(folder, i))

    if len(gpus) > 1:
        print("You are using 2 GPUs while the code is set up for one only.")
        exit()

    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=opt,
        metrics='accuracy',
        student_loss_fn=loss,
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=10,
    )

    metrics_callback = NewCallback(val_dataset, val_filenames)

    distiller.fit(train_dataset,
            validation_data=val_dataset,
            verbose=2,
            epochs=parameters.n_epochs,
            # class_weight=weights,
            callbacks=[metrics_callback]
    )

    for i, sp2 in enumerate(s):
        sp2 = np.expand_dims(sp2, axis=0)
        sp2 = student(sp2, return_spec=True)
        sp2 = np.swapaxes(np.squeeze(sp2.numpy()), 0, 1)
        visualize_spec_bis(sp2, sr=parameters.sr, dest="{}/backend_after_{}".format(folder, i))

        diff = sp2-sps[i]
        visualize_spec_bis(diff, sr=parameters.sr, dest="{}/backend_diff_{}".format(folder, i))
    

    # # training frontend
    # for l in model.layers:
    #     if l.name == "leaf":
    #         l.trainable = True
    #     else:
    #         l.trainable = False

    # # parameters.lr = 1e-4
    # # parameters.n_epochs = 5
    # # opt = tf.keras.optimizers.Adam(
    # #     lr=parameters.lr, beta_1=0.9, beta_2=0.999, epsilon=parameters.epsilon, decay=parameters.weight_decay, amsgrad=False
    # # )
    # optimizers = [
    #     tf.keras.optimizers.Adam(
    #         lr=parameters.lr, beta_1=0.9, beta_2=0.999, epsilon=parameters.epsilon, decay=parameters.weight_decay, amsgrad=False
    #     ),
    #     tf.keras.optimizers.Adam(
    #         lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=parameters.epsilon, decay=parameters.weight_decay, amsgrad=False
    #     )
    # ]
    # optimizers_and_layers = [(optimizers[0], model.layers[0]), (optimizers[1], model.layers[1:])]
    # opt = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    
    # model.compile(
    #     optimizer=opt,
    #     loss=loss,
    #     metrics=[
    #         'accuracy', 
    #     ],
    # )

    # # folder = "{}/others".format(parameters.job_dir)
    # sps = []
    # s = list(train_dataset.as_numpy_iterator())[0][0][:3]
    # for i, sp in enumerate(s):
    #     sp = np.expand_dims(sp, axis=0)
    #     sp = model(sp, return_spec=True)
    #     sp = np.swapaxes(np.squeeze(sp.numpy()), 0, 1)
    #     sps.append(sp)
    #     visualize_spec_bis(sp, sr=parameters.sr, dest="{}/frontend_before_{}".format(folder, i))

    # if len(gpus) > 1:
    #     print("You are using 2 GPUs while the code is set up for one only.")
    #     exit()

    # # training
    # model.fit(
    #     train_dataset,
    #     epochs=parameters.n_epochs,
    #     verbose=2,
    #     class_weight=weights,
    #     callbacks=[metrics_callback]
    # )

    # for i, sp2 in enumerate(s):
    #     sp2 = np.expand_dims(sp2, axis=0)
    #     sp2 = model(sp2, return_spec=True)
    #     sp2 = np.swapaxes(np.squeeze(sp2.numpy()), 0, 1)
    #     visualize_spec_bis(sp2, sr=parameters.sr, dest="{}/frontend_after_{}".format(folder, i))

    #     diff = sp2-sps[i]
    #     visualize_spec_bis(diff, sr=parameters.sr, dest="{}/frontend_diff_{}".format(folder, i))

    # model.save(parameters.job_dir + "/model_{}.h5".format(parameters.n_epochs))

def launch_job(datasets, model, spec_aug_params, audio_aug_params, parse_function):
    '''
    parameters: 
    # dictionary:  datasets to use, 
    # model:  imported from modules/models.py, 
    # augmentation parameters:  (No augmentation if empty list is passed), 
    # parse function:  (inside modules/parsers) used to create tf.dataset object in create_tf_dataset
    '''
    # in a given file named train$ (parent/cache folder named train$), we can have multiple jobs (child folders named 1,2,3)
    initialize_job() #  initialize each (child) job inside the file (i.e, creates all the subfolders like tp/tn/gradcam/etc, file saving conventions, etc)
    train_model(datasets, model, spec_aug_params, audio_aug_params, parse_function)

if __name__ == "__main__":
    
    print("Tensorflow Version: {}".format(tf.__version__))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    seed_everything() # seeding np, tf, etc
    arguments = parameters.parse_arguments()
    parameters.init()
    parameters.mode = "pneumonia" if arguments["mode"] == "main" else arguments["mode"]
    parameters.n_classes = 2 if parameters.mode == "cw" else 1
    print(parameters.cache_root)
    print(parameters.mode)
    print(os.path.basename(__file__).split('.')[0])
    parameters.file_dir = os.path.join(parameters.cache_root, parameters.mode, os.path.basename(__file__).split('.')[0])
    parameters.description = arguments["description"]
    testing_mode(int(arguments["testing"])) # if true, adds _testing to the cache folder, sets a small number of epochs, smaller dataset, turn off a few settings in the callback, etc
    # works to set up the parent folder (i.e., overwrites if already exists) + works well with initialize_job above
    initialize_file_folder()
    print("-----------------------")


    ###### set up used for spec input models (default)
    parameters.hop_length = 254
    parameters.shape = (128, 311)
    parameters.n_sequences = 9
    spec_aug_params = []
    audio_aug_params = []
    launch_job({"Bd": 0, "Jordan": 0, "Icbhi": 1, "Perch": 1, "Ant": 1, "SimAnt": 1,}, leaf_model9_model, spec_aug_params, audio_aug_params, None)
    
    # to run another job, add a line to modify whatever parameters, and rerun a launch_job function as many times as you want!
    # parameters.hop_length = 512
    # launch_job({"Bd": 0, "Jordan": 1, "Icbhi": 1, "Perch": 0, "Ant": 0, "SimAnt": 0,}, model9, spec_aug_params, audio_aug_params, spec_parser)
