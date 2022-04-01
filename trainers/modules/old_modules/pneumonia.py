import pandas as pd
import xlwt 
from xlwt import Workbook 
import glob
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, precision_recall_curve, auc
# BANGLADESH EXCEL FUNCTION
    # train_number = 
    # excel_path = "/home/alirachidi/classification_algorithm/data/Bangladesh_PCV_onlyStudyPatients.xlsx"
    # excel_dest = "/home/alirachidi/classification_algorithm/data/excel_spreadsheets/validation_{}.xls".format(file_number)

def process_icbhi(six, dataset_path):
    _list = os.path.join(dataset_path, 'aa_paths_and_labels.txt')
    filenames = []
    labels = []
    with open(_list) as infile:
        for line in infile:
            elements = line.rstrip("\n").split(',')
            # print(elements)
            filenames.append(elements[0])
            labels.append((int(elements[1])))
    return filenames, labels

def process_bangladesh(six, dataset_path, root):

    # root = "../../data/PCV_SEGMENTED_Processed_Files/" 
    excel_path = "/home/alirachidi/classification_algorithm/data/Bangladesh_PCV_onlyStudyPatients.xlsx"

    filenames = []
    labels = []
    df = pd.read_excel(excel_path, engine='openpyxl')

    folder_paths = glob.glob(os.path.join(root, "*"),recursive=True)
    print("Original number of folders: {}".format(len(folder_paths)))
    folder_paths = sorted(folder_paths)

    count_1 = 0
    count_2 = 0
    count_3 = 0

    for folder_path in folder_paths:
        if folder_path == os.path.join(root, "0365993_SEGMENTED") or folder_path == os.path.join(root, "0273320_SEGMENTED") or folder_path == os.path.join(root, "0364772_SEGMENTED"):
            # print("here")
            continue
        recordings = glob.glob(os.path.join(folder_path, "*.wav"))
        recordings = sorted(recordings)
        # print(recordings)
        # exit()
        if len(recordings) != 6:
            # print("recordings not equal 6")
            count_1 += 1
            continue
        patient_id = int(folder_path.split('/')[-1].split('_')[0])
        file_column = df.loc[df['HOSP_ID'] == patient_id]
        if file_column.empty:
            # print("empty col")
            count_2 += 1
            continue
        label = str(file_column['PEP1'].values[0])
        if label == "Uninterpretable":
            count_3 += 1
            continue
        final_chunks = []
        for recording in recordings:
            recording_name = recording.split('/')[-1].split('.')[0]
            # print(recording_name)
            chunks = sorted(glob.glob(os.path.join(dataset_path, "{}*.txt".format(recording_name))))
            # print(chunks)
            # exit()
            if six:
                # if len(chunks) > 1:
                final_chunks.append(chunks[int(len(chunks)/2)])
                # else:
                    # final_chunks.append(chunks[0])
            else:
                final_chunks.extend(chunks)
        filenames.append(final_chunks)
        labels.append(convert_pneumonia_label(label))
    
    # print("count_1:{}".format(count_1))
    # print("count_2:{}".format(count_2))
    # print("count_3:{}".format(count_3))
    # print(count_1)
    # print(count_1)

    return filenames, labels

def process_data(val_samples, train_samples, parse_func, shape, batch_size, initial_channels, cuberooting, normalizing):
    
    train_filenames, train_labels = zip(*train_samples)
    val_filenames, val_labels = zip(*val_samples)

    train_dataset = tf.data.Dataset.from_tensor_slices((list(train_filenames), list(train_labels)))
    train_dataset = train_dataset.shuffle(len(train_filenames))
    train_dataset = train_dataset.map(lambda filename, label: parse_func(filename, label, shape, initial_channels, cuberooting, normalizing), num_parallel_calls=4)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(1)

    val_dataset = tf.data.Dataset.from_tensor_slices((list(val_filenames), list(val_labels)))
    val_dataset = val_dataset.shuffle(len(val_filenames))
    val_dataset = val_dataset.map(lambda filename, label: parse_func(filename, label, shape, initial_channels, cuberooting, normalizing), num_parallel_calls=4)
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(1)

    return val_dataset, train_dataset


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

def return_best_six(filename):

    # print(filename)
    all_merged = []
    merged = []
    for i in range(0, len(filename)): 
        f = filename[i]
        index = int(f.split('.')[-2][-1])
        # print(index)
        if index == 0:
            if i == 0:
                merged.append(f)
                continue
            # print("here: {}".format(merged))
            all_merged.append(merged)
            merged = []
            merged.append(f)
        else:
            merged.append(f)
        if i == len(filename) - 1:
                all_merged.append(merged)
        # f = f.split('.')[0][-1]
    # print(all_merged)
    best_six = [merged[int(len(merged)/2)] for merged in all_merged]
    # print(best_six)
    return best_six

def generate_bangladesh_sheet(model, shape, grouped_val_samples, excel_dest, six, initial_channels, pneumonia_threshold):

    print("Starting...")
    wb = Workbook() 
    sheet1 = wb.add_sheet('Sheet_1', )

    row = 1
    col = 0
    true_positives = 0
    sheet1.write(0, 0, "Patient ID") 
    sheet1.write(0, 1, "File name") 
    sheet1.write(0, 2, "Body area level decision") 
    sheet1.write(0, 3, "Body area level decision (count)") 
    sheet1.write(0, 4, "Patient level decision") 
    sheet1.write(0, 5, "Label") 

    y_true = []
    y_pred = []
    for i, sample in enumerate(grouped_val_samples):
            # if i == 0:
            #     continue
            filename = sample[0]
            label = sample[1]
            name = filename[0].split('/')[-1][:9]
            if not six:
                # do some processing to get the best 6
                filename = return_best_six(filename)
            counter = 0
            # file_column = df.loc[df['HOSP_ID'] == patient_id]
            for extract in filename:
                sheet1.write(row, 0, name) 
                sheet1.write(row, 1, extract.split('/')[-1]) 
                arr = np.loadtxt(extract, delimiter=',')
                arr = np.repeat(arr[..., np.newaxis], initial_channels, -1)
                if len(shape) == 4:
                    arr = arr[:, :, 0]
                    arr = np.transpose(arr)
                    arr = np.reshape(arr, newshape=shape)
                    # print(arr.shape)
                output = model.predict(np.array([arr]))
                output = round(output[0][0])
                if output == 1:
                    counter += 1
                sheet1.write(row, 2, inverse_convert_pneumonia_label(output))
                row += 1
            # calculating accuracy 
            if ((counter >= pneumonia_threshold) and (label == 1)) or ((counter == 0) and (label == 0)):
                true_positives += 1 
            sheet1.write(row, 3, counter)
            # setting prediction
            if counter >= pneumonia_threshold:
                sheet1.write(row, 4, "PEP")
                y_pred.append(1)
            else:
                sheet1.write(row, 4, "NO PEP")
                y_pred.append(0)
            y_true.append(label)
            sheet1.write(row, 5, inverse_convert_pneumonia_label(label))
            row += 1
 
    cm = confusion_matrix(y_true, y_pred)
    acc = np.trace(cm) / np.sum(cm)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='binary')

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    area_under_curve = auc(recalls, precisions)

    print("CM: {}".format(cm))
    print("F1: {}".format(f1))
    print("AUC: {}".format(area_under_curve))
    print("Accuracy: {}".format(acc))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("Ended. Saving the Excel Sheet {}...".format(excel_dest))
    wb.save(excel_dest) 
    print("Done.")
    return f1, acc, precision, recall, area_under_curve, y_true, y_pred