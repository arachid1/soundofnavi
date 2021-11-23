from ..main import parameters
from ..main.global_helpers import visualize_spec
# from .helpers import *
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

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, precision_recall_curve, auc, plot_precision_recall_curve, PrecisionRecallDisplay, roc_curve, plot_roc_curve, RocCurveDisplay
import itertools
import librosa
from librosa import display
import soundfile as sf
import wandb

np.set_printoptions(precision=3)

class CustomMetricsCallback3(tf.keras.callbacks.LambdaCallback):

    def __init__(self, validation_data, val_filenames, shape, initial_channels, n_classes, job_dir, sr, min_delta, es_patience, lr_patience, min_lr, factor, epoch_start, target, job_count, clause, samples_to_vis, parse_function, cuberooting, normalizing, adaptive_lr=True, average_mode='binary', clause_portion=0.9):

        self.validation_data = validation_data
        self.val_filenames = val_filenames
        self.visualize_spec = types.MethodType(visualize_spec, self)
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
        self.best_roc_auc = 0
        self.best_avg_accuracy = 0
        self.best_cm = 0
        self.best_patient_cm = 0
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
        tf.print("Best ROC auc: {}".format(self.best_roc_auc))
        tf.print("Best average accuracy: {}".format(self.best_avg_accuracy))
        tf.print("Best patient cm: {}".format(self.best_patient_cm))
        tf.print("Best accuracy: {:.2f}".format(self.best_accuracy))
        tf.print("Best recall: {:.2f}".format(self.best_recall))
        tf.print("Best precision: {:.2f}".format(self.best_precision))
        tf.print("Best AUC: {:.2f}".format(self.best_auc))
        tf.print("Best class accuracies: {}".format(self.best_class_accuracies))
        tf.print("Best epoch: {}".format(self.best_epoch))

        # wandb.run.summary.update({"best_val_f1": self.best_f1})
        # wandb.run.summary.update({"best_val_accuracy": self.best_accuracy})
        # wandb.run.summary.update({"best_val_precision": self.best_precision})
        # wandb.run.summary.update({"best_val_recall": self.best_recall})
        # wandb.run.summary.update({"best_auc": self.best_auc})
        # wandb.run.summary.update({"best_epoch": self.best_epoch})

    def on_epoch_end(self, epoch, logs=None):

        one_indexed_epoch = epoch + 1

        cm = np.zeros((self.n_classes + 1, self.n_classes + 1))

        y_true_all = []
        y_pred_all = []
        preds_all = []
        x_all = []
    
        patients_dict = defaultdict(list) # struct is { patient_id: [[name, y_pred, y_true, tn, fn, fp, tp]] }
        fns = []
        fps = []
        tps = []

        # print("inside cb")
        
        for i, (batch, y_true) in enumerate(self.validation_data):
            batch = batch.numpy()
            y_true = y_true.numpy()
            filename = self.val_filenames[i]
            if filename.startswith("0"):
                patient_id = filename[:8]
            else:
                patient_id = filename.split('_')[0]
            preds = self.model.predict(batch)
            preds = preds.reshape(len(preds))
            y_pred = np.zeros(preds.shape)
            y_pred[preds >= 0.5] = 1
            y_pred[preds < 0.5] = 0

            batch = np.squeeze(batch)
            if y_true == 0 and y_pred == 0: # tn
                patients_dict[patient_id].append([filename, preds, y_true, 1, 0, 0, 0])
            if y_true == 1 and y_pred == 0: # fn
                dest = os.path.join(self.job_directory, "fn", filename)
                fns.append([batch, dest, preds, y_true])
                patients_dict[patient_id].append([filename, preds, y_true, 0, 1, 0, 0])
            if y_true == 0 and y_pred == 1: # fp
                dest = os.path.join(self.job_directory, "fp", filename)
                fps.append([batch, dest, preds, y_true])
                patients_dict[patient_id].append([filename, preds, y_true, 0, 0, 1, 0])
            if y_true == 1 and y_pred == 1: # tp
                dest = os.path.join(self.job_directory, "tp", filename)
                tps.append([batch, dest, preds, y_true])
                patients_dict[patient_id].append([filename, preds, y_true, 0, 0, 0, 1])
            preds_all.extend(preds)
            y_true_all.append(y_true)
            y_pred_all.append(y_pred)
            x_all.append(batch)


        # ROC AUC
        fpr, tpr, thresholds = roc_curve(y_true_all, preds_all)
        roc_area_under_curve = auc(fpr, tpr)

        # AUC & PR Curve
        # roc_display = plot_roc_curve(self.model, x_all, y_true_all)
        # plt.xticks(np.arange(0, 1, 0.1))
        # plt.yticks(np.arange(0, 1, 0.1))
        # plt.close()
        # RocCurveDisplay.plot()
        # pr_display = RocCurveDisplay(fpr, tpr, roc_auc=roc_area_under_curve)
        # pr_display = pr_display.plot()
        # pr_display.ax_.get_legend().remove()
        # pr_display.ax_.xaxis.set_ticks(np.arange(0, 1, 0.1))
        # pr_display.ax_.yaxis.set_ticks(np.arange(0, 1, 0.1))
        # path = os.path.join(self.job_directory, "test_pr_curve_{}.png".format(one_indexed_epoch))
        # path = "pr_curves/test_pr_curve_{}.png".format(one_indexed_epoch)


        # CM, accuracy, precision, recall, f1 and class accuracies
        cm = confusion_matrix(y_true_all, y_pred_all)
        normalized_cm = confusion_matrix(y_true_all, y_pred_all, normalize='true')
        cm_sum = np.sum(cm)
        acc = np.trace(cm) / cm_sum
        class_accuracies=self.compute_class_accuracies(cm)

        precision, recall, f1, support = precision_recall_fscore_support(y_true_all, y_pred_all, average=self.average_mode)

        avg_accuracy = np.sum(class_accuracies) / len(class_accuracies)
        
        tf.print("Confusion matrix: \n {}".format(cm))
        tf.print("Normalized Confusion matrix: \n {}".format(normalized_cm))
        tf.print("Validation f1: {:.2f}".format(f1*100))
        tf.print("Validation ROC auc: {}".format(roc_area_under_curve))
        tf.print("Validation average accuracy: {}".format(avg_accuracy))
        tf.print("Validation accuracy: {:.2f}".format(acc*100))
        tf.print("Validation recall: {:.2f}".format(recall*100))
        tf.print("Validation precision: {:.2f}".format(precision*100))
        # tf.print("Validation AUC: {}".format(area_under_curve))
        tf.print("Validation class accuracies: {}".format(class_accuracies))

        # wandb.log({'train_accuracy': logs['accuracy']})
        # wandb.log({'train_loss': logs['loss']})
        # wandb.log({'val_recall': recall})
        # wandb.log({'val_precision': precision})
        # wandb.log({'val_f1': f1})
        # # wandb.log({'val_auc': area_under_curve})
        # wandb.log({'lr': self.model.optimizer.lr})

        # print("TARGET: {}".format(self.target))
        # if self.target == 0: # auc 
        #     target_metric = area_under_curve
        #     # print("target metric(test auc): {}".format(target_metric))
        #     best_target_metric = self.best_auc
        # elif self.target == 1: # f1
        #     target_metric = f1
        #     # print("target metric(test f1): {}".format(target_metric))
        #     best_target_metric = self.best_f1

        # target_metric = roc_area_under_curve
        # best_target_metric = self.best_roc_auc

        target_metric = avg_accuracy
        best_target_metric = self.best_avg_accuracy
        
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
        if ((one_indexed_epoch >= self.epoch_start) and (target_metric > best_target_metric)) or parameters.testing:
            "New best results were achieved."
            self.best_model = self.model
            self.best_f1 = f1
            self.best_roc_auc = roc_area_under_curve
            self.best_avg_accuracy = avg_accuracy
            self.best_cm = cm
            self.best_normalized_cm = normalized_cm
            # plt.savefig()
            # self.best_auc = area_under_curve
            self.best_accuracy = acc
            self.best_precision = precision
            self.best_recall = recall
            self.best_class_accuracies = class_accuracies
            self.best_epoch = one_indexed_epoch
            self.best_patient_cm = self.generate_patient_report(patients_dict)
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
            # for fn in fns:
            #     self.visualize_spec(fn[0], 8000, fn[1], title = str(fn[2]) + "__" + str(fn[3]))
            # for fp in fps:
            #     self.visualize_spec(fp[0], 8000, fp[1], title = str(fp[2]) + "__" + str(fp[3]))
            # for tp in tps:
            #     self.visualize_spec(tp[0], 8000, tp[1], title = str(tp[2]) + "__" + str(tp[3]))
            
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
            if self.tracker == self.es_patience:
                print("The number of epochs since last 1% equals the patience")
                self.model.stop_training = True
            else:
                print("The validation tracker metric at {} hasn't increased by {} in {} epochs".format(self.tracker_metric, self.min_delta, self.tracker))
        return None


    def generate_patient_report(self, patients_dict):
        tn_path = self.job_dir + "/{}/tns.txt".format(self.job_count)
        report_path = self.job_dir + "/{}/report.txt".format(self.job_count)
        tn_writer = open(tn_path, "w")
        reports_writer = open(report_path, "w")
        y_true = []
        y_pred = []
        for patient_id, patient_recordings in patients_dict.items():
            tn = 0
            fn = 0
            fp = 0
            tp = 0
            for recording in patient_recordings:
                tn += recording[3]
                fn += recording[4]
                fp += recording[5]
                tp += recording[6]
            label = patient_recordings[0][2]
            if fp > 0 or tp > 0:
                pred_class = 1
            else:
                pred_class = 0
            y_true.append(label)
            y_pred.append(pred_class)
            status = self.return_status(label, pred_class)
            to_write = "{}, a {} patient, has {} tns, {} fns, {} fps, and {} tps out of {} recordings\n".format(patient_id, status, tn, fn, fp, tp, len(patient_recordings))
            if (label == 0) and (pred_class == 0):
                tn_writer.write(to_write)
            else:
                reports_writer.write(to_write)
        
        cm = confusion_matrix(y_true, y_pred)
        normalized_cm = confusion_matrix(y_true, y_pred, normalize='true')
        cm_sum = np.sum(cm)
        acc = np.trace(cm) / cm_sum
        class_accuracies=self.compute_class_accuracies(cm)
        message = "\nPatient Confusion matrix: \n {} \n".format(cm)
        tf.print(message)
        reports_writer.write(message)
        message = "Patient Normalized Confusion matrix: \n {} \n".format(normalized_cm)
        tf.print(message)
        reports_writer.write(message)
        message = "Patient Validation accuracy: \n {:.2f} \n".format(acc*100)
        tf.print(message)
        reports_writer.write(message)
        message = "Patient Validation class accuracies: \n {} \n".format(class_accuracies)
        tf.print(message)
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