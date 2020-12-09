import json
import argparse
import os
from tensorflow.keras import backend as K
import random
import wave
import math
import scipy.io.wavfile as wf
import sys
from scipy.signal import lfilter
from audiomentations import *
import subprocess
import pickle
import unittest
from librosa import display
import librosa
from decimal import *
from sklearn.model_selection import train_test_split
import importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


K.clear_session()

def seed_everything():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(0)
    random.seed(0)

def write_record(params, file_path):

    file_name = file_path.split('/')[-1]
    date = datetime.datetime.now()
    day = datetime.datetime.today().weekday()
    time = date.time()
    year = date.isocalendar()[0]
    week = date.isocalendar()[1]
    record_path = str('records/' + str(week) + '_' + str(year) + '_record.txt')
    with open(record_path , 'a') as f:
        f.write(str(day) + "  " + str(time) + '\n')
        f.write(file_name + '\n')
        f.write(str(params) + '\n')
        f.write('\n')

def print_sample_count(src_dict):

    print('all:{}\nnone:{}\ncrackles:{}\nwheezes:{}\nboth:{}'.format(
        len(src_dict['none']) + len(src_dict['crackles']) +
        len(src_dict['wheezes']) + len(src_dict['both']),
        len(src_dict['none']),
        len(src_dict['crackles']),
        len(src_dict['wheezes']),
        len(src_dict['both'])))


def Extract_Annotation_Data(file_name, root):
    tokens = file_name.split('_')
    recording_annotations = pd.read_csv(os.path.join(
        root, file_name + '.txt'), names=['Start', 'End', 'Crackles', 'Wheezes'], delim_whitespace=True)
    return recording_annotations


def read_wav_file(str_filename, target_rate):
    wav = wave.open(str_filename, mode='r')

    (sample_rate, data) = extract2FloatArr(wav, str_filename)

    if (sample_rate != target_rate):
        # print("is resampling...")
        (_, data) = resample(sample_rate, data, target_rate)
    return (target_rate, data.astype(np.float32))


def resample(current_rate, data, target_rate):
    x_original = np.linspace(0, 100, len(data))
    new_length = int(len(data) * (target_rate / current_rate))
    x_resampled = np.linspace(0, 100, new_length)
    resampled = np.interp(x_resampled, x_original, data)
    return (target_rate, resampled.astype(np.float32))
# -> (sample_rate, data)


def extract2FloatArr(lp_wave, str_filename):
    (bps, channels) = bitrate_channels(lp_wave)

    if bps in [1, 2, 4]:  # d epth
        (rate, data) = wf.read(str_filename)
        divisor_dict = {1: 255, 2: 32768}
        if bps in [1, 2]:
            divisor = divisor_dict[bps]
            data = np.divide(data, float(divisor))  # clamp to [0.0,1.0]
        return (rate, data)

    elif bps == 3:
        # 24bpp wave
        return read24bitwave(lp_wave)

    else:
        raise Exception(
            'Unrecognized wave format: {} bytes per sample'.format(bps))

# Note: This function truncates the 24 bit samples to 16 bits of precision
# Reads a wave object returned by the wave.read() method
# Returns the sample rate, as well as the audio in the form of a 32 bit float numpy array
# (sample_rate:float, audio_data: float[])


def read24bitwave(lp_wave):
    nFrames = lp_wave.getnframes()
    buf = lp_wave.readframes(nFrames)
    reshaped = np.frombuffer(buf, np.int8).reshape(nFrames, -1)
    short_output = np.empty((nFrames, 2), dtype=np.int8)
    short_output[:, :] = reshaped[:, -2:]
    short_output = short_output.view(np.int16)
    # return numpy array to save memory via array slicing
    return (lp_wave.getframerate(), np.divide(short_output, 32768).reshape(-1))


def bitrate_channels(lp_wave):
    bps = (lp_wave.getsampwidth() / lp_wave.getnchannels())  # bytes per sample
    return (bps, lp_wave.getnchannels())


def slice_icbhi_data(start, end, raw_data,  sample_rate, max_ind):
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)
    return raw_data[start_ind: end_ind], start_ind, end_ind


def slice_perch_data(start, end, raw_data, sample_rate):
    start_ind = int(start * sample_rate)
    end_ind = int(end * sample_rate)
    return raw_data[start_ind: end_ind]


def process_overlaps(overlaps):
    crackles = sum([overlap[2] for overlap in overlaps])
    wheezes = sum([overlap[3] for overlap in overlaps])
    crackles = 1.0 if (not(crackles == 0)) else 0.0
    wheezes = 1.0 if (not(wheezes == 0)) else 0.0
    return crackles, wheezes

def evaluate_imbalance(column):
    neg, pos = np.bincount(column)
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))

def visualize_spec(data, sr):
    
    fig = plt.figure(figsize=(20, 10))
    display.specshow(
        data,
        # y_axis="log",
        sr=sr,
        cmap="coolwarm"
    )
    plt.colorbar()
    plt.show()


def find_labels(times, start, end, overlap_threshold, audio_length, sample_rate):
    overlaps = []
    for time in times:
        if start > time[0]:  # the window starts after
            if start >= time[1]:  # no overlap
                continue
            if end <= time[1]:  # perfect overlap: window inside breathing pattern
                # shouldn't happen since window is 2 sec long
                if (time[1] - time[0]) < (overlap_threshold * audio_length * sample_rate):
                    continue
                else:
                    return time[2], time[3]
            else:  # time[1] < end: partial overlap, look at next window c)
                if (time[1] - start) < (overlap_threshold * audio_length * sample_rate):
                    continue
                else:
                    overlaps.append(time)
        else:  # the window starts before
            if end <= time[0]:  # no overlap
                continue
            if end < time[1]:  # partial overlap b)
                if (end - time[0]) < (overlap_threshold * audio_length * sample_rate):
                    continue
                else:
                    overlaps.append(time)
            else:  # end > time[1]: perfect overlap: breathing pattern inside of window
                # shouldn't happen since all windows are 1 sec or more
                if (time[1] - time[0]) < (overlap_threshold * audio_length * sample_rate):
                    continue
                else:
                    overlaps.append(time)
    return process_overlaps(overlaps)


def stretch(original, sample_rate, audio_length):
    stretch_amount = 1 + (len(original) / (audio_length * sample_rate))
    (_, stretched) = resample(sample_rate,
                              original, int(sample_rate * stretch_amount))
    return stretched


def generate_padded_samples(source, output_length):
    output_length = int(output_length)
    copy = np.zeros(output_length, dtype=np.float32)
    src_length = len(source)
    frac = src_length / output_length
    if(frac < 0.5):
        # tile forward sounds to fill empty space
        cursor = 0
        while(cursor + src_length) < output_length:
            copy[cursor:(cursor + src_length)] = source[:]
            cursor += src_length
    else:
        copy[:src_length] = source[:]
    #
    return copy


def find_times(recording_annotations, sample_rate, first):
    times = []
    for i in range(len(recording_annotations.index)):
        row = recording_annotations.loc[i]
        c_start = Decimal('{}'.format(row['Start']))
        c_end = Decimal('{}'.format(row['End']))
        if (c_end - c_start < 1):
            continue
        if (first):
            rw_index = c_start
            first = False
        times.append((c_start * sample_rate, c_end * sample_rate,
                      row['Crackles'], row['Wheezes']))  # get all the times w labels
    return times, rw_index


def write_to_file(data, name, path):
    output = open(path, 'wb')
    pickle.dump(data, output)
    if name is not None:
        print("Find {} dataset at {}".format(name, path))
    output.close()


def split(no_labels, c_only, w_only, c_w, train_test_ratio):
    none_train, none_test = train_test_split(
        no_labels, test_size=train_test_ratio)
    c_train, c_test = train_test_split(c_only, test_size=train_test_ratio)
    w_train, w_test = train_test_split(w_only, test_size=train_test_ratio)
    c_w_train, c_w_test = train_test_split(c_w, test_size=train_test_ratio)

    train_dict = {'none': none_train, 'crackles': c_train,
                  'wheezes': w_train, 'both': c_w_train}

    test_dict = {'none': none_test, 'crackles': c_test,
                 'wheezes': w_test, 'both': c_w_test}

    return train_dict, test_dict

def generate_cochlear_spec(x, cochlear_parameters, coch_b, coch_a, order):

    # get filter bank
    L, M = coch_b.shape
    # print("Shape: [{}, {}]".format(L, M))
    L_x = len(x)

    # octave shift, nonlinear factor, frame length, leaky integration
    shft = cochlear_parameters["shft"] #octave shift (Matlab index: 4)
    fac = cochlear_parameters["fac"] # nonlinear factor (Matlab index: 3)
    frmlen = cochlear_parameters["frmlen"] # (Matlab index: 1)
    L_frm = np.round(frmlen * (2**(4+shft)))   #frame length (points)
    tc = cochlear_parameters["tc"] # time constant (Matlab index: 2)
    tc = np.float64(tc)
    alph_exponent = -(1/(tc*(2**(4+shft))))
    alph = np.exp(alph_exponent) # decaying factor
    
    # get data, allocate memory for output
    N = np.ceil(L_x/L_frm) # number of frames
    # print("Number of frames: {}".format(N))
    # x = generate_padded_samples(x, ) # TODO: come back to padding
    
    v5 = np.zeros([int(N), M - 1])
    
    ##############################
    # last channel (highest frequency)
    ############################## 

    # get filters from stored matrix
    # print("Number of filters: {}".format(M))
    p = int(order[M-1]) # ian's change 11/6

    # ian changed to 0 because of the way p, coch_a, and coch_b are seperated
    B =  coch_b[0:p+1, M - 1] # M-1 before
    A =  coch_a[0:p+1, M - 1]
    y1 = lfilter(B, A, x)
    y2 = y1
    y2_h = y2
    
    # All other channels
    for ch in range(M-2, -1, -1):

        # ANALYSIS: cochlear filterbank
        # IIR: filter bank convolution ---> y1  
        p = int(order[ch])
        B =  coch_b[0:p+1, ch]
        A =  coch_a[0:p+1, ch]

        y1 = lfilter(B, A, x)

        # TRANSDUCTION: hair cells
        y2 = y1   
        
        # REDUCTION: lateral inhibitory network 
        # masked by higher (frequency) spatial response
        y3 = y2 - y2_h
        y2_h = y2
        
        # half-wave rectifier 
        y4 = y3.copy()
        y4[y4 < 0] = 0
            
        # temporal integration window #
        # leaky integration. alternative could be simpler short term average
        y5 = lfilter([1], [1-alph], y4)
        
        v5_row = []
        for i in range(1, int(N) + 1):
            v5_row.append(y5[int(L_frm*i) - 1])
        v5[:, ch] = v5_row
        # v5[:, ch] = y5[int(L_frm):int(L_frm*N)] # N elements, each space with L_frm
    # print("...End")
    
    return v5
    

def convert_to_spec(data, spec_type, sr, n_fft, hop_length, spec_win_length, n_mels):

    n_fft = n_fft
    hop_length = hop_length
    spec_list = []
    if spec_type == "log":
        spec_win_length = spec_win_length
        for d in data:
            log_spectrogram = librosa.power_to_db(
                np.abs(librosa.stft(d[0], n_fft=n_fft, hop_length=hop_length, win_length=spec_win_length,
                                    center=True, window='hann')) ** 2, ref=1.0)
            spec_list.append((log_spectrogram, d[1], d[2], d[0]))
    elif spec_type == "coch":
        for d in data:
            fs = 8000 #changed by ian 11/6 from a mistake in runme.txt
            bp = 1
            cochlear_parameters = {"frmlen": 8*(1/(fs/8000)), "tc": 8, "fac": -2, "shft": np.log2(fs/16000), "FULLT": 0, "FULLX": 0, "bp": bp}
            coch_path = '/Users/alirachidi/Documents/Sonavi_Labs/classification_algorithm/cochlear_preprocessing/'
            coch_a = np.loadtxt(coch_path + 'COCH_A.txt', delimiter=',')
            coch_b = np.loadtxt(coch_path + 'COCH_B.txt', delimiter=',')
            order = np.loadtxt(coch_path + 'p.txt', delimiter=',')
            coch_spectrogram = generate_cochlear_spec(d[0], cochlear_parameters, coch_b, coch_a, order)
            coch_spectrogram = np.transpose(coch_spectrogram)
            # print(coch_spectrogram.shape)å
            spec_list.append((coch_spectrogram, d[1], d[2], d[0]))
            if len(spec_list) % 2000 == 0:
                print("Completed {} elements...".format(len(spec_list)))
    else:  # TODO: change it to regular if statement
        for d in data:
            n_mels = n_mels
            mel_spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(
                d[0], sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels), ref=np.max)
            spec_list.append((mel_spectrogram, d[1], d[2], d[0]))
    return spec_list


def gen_time_stretch(original, sample_rate, max_percent_change):
    stretch_amount = 1 + np.random.uniform(-1, 1) * (max_percent_change / 100)
    (_, stretched) = resample(sample_rate,
                              original, int(sample_rate * stretch_amount))
    return stretched


def time_stretch(data, target, sr, max_percent_change):
    nb_augmentations = target - len(data)
    print("len(data):" + str(len(data)))
    print("nb_augmentations:" + str(nb_augmentations))
    augmented = data.copy()
    cycles_with_labels = []
    for i in range(0, nb_augmentations):
        source = data[i % len(data)]
        stretched = gen_time_stretch(source[0], sr, max_percent_change)
        cycles_with_labels.append((stretched, source[1], source[2]))
    augmented.extend(cycles_with_labels)
    print("len(augmented):" + str(len(augmented)))
    return augmented


def get_icbhi_samples(recording_annotations, file_name, root, sample_rate, file_id, overlap_threshold=0.15, audio_length=2, step_size=1):
    sample_data = [file_name, file_id]
    (rate, data) = read_wav_file(os.path.join(
        root, file_name + '.wav'), sample_rate)
    times, rw_index = find_times(recording_annotations, sample_rate, True)
    max_ind = int(times[len(times) - 1][1])
    step = True
    count = 0
    while step:
        count += 1
        start = rw_index
        end = rw_index + audio_length
        rw_audio_chunk, start_ind, end_ind = slice_icbhi_data(
            start, end, data, rate, max_ind)
        if (not (abs(end_ind - start_ind) == (audio_length * sample_rate))):  # ending the loop
            step = False
            if (abs(start_ind - end_ind) < (sample_rate)):  # disregard if less than one second
                continue
            else:  # 0 pad if more than one second
                rw_audio_chunk = generate_padded_samples(
                    rw_audio_chunk, sample_rate * audio_length)
        crackles, wheezes = find_labels(
            times, start_ind, end_ind, overlap_threshold, audio_length, sample_rate)
        sample_data.append((rw_audio_chunk, crackles, wheezes))
        rw_index = start + step_size
    return sample_data


def get_perch_sample(recording_annotations, file_name, root, sample_rate, file_id, audio_length):
    sample_data = [file_name, file_id]
    (rate, data) = read_wav_file(os.path.join(
        root, file_name + '.wav'), sample_rate)
    row = recording_annotations.loc[0]
    c_start = Decimal('{}'.format(row['Start']))
    c_end = Decimal('{}'.format(row['End']))
    rw_audio_chunk = slice_perch_data(c_start, c_end, data, rate)
    if len(rw_audio_chunk) < sample_rate * audio_length:
        rw_audio_chunk = generate_padded_samples(
            rw_audio_chunk, sample_rate * audio_length)
    sample_data.append((rw_audio_chunk, row['Crackles'], row['Wheezes']))
    return sample_data


def generate_icbhi(root, sr, overlap_threshold, audio_length, step_size, augmentation, train_test_ratio, height, width, spec_type, n_fft, hop_length, spec_win_length, n_mels, tests_destination, test_files_destination, test):
    print("Generating ICBHI...")
    # print("Directory: {}.".format(os.listdir(root)))
    filenames = [s.split('.')[0]
                 for s in os.listdir(path=root) if '.txt' in s]
    # ASSERT
    print("Number of Files: {}".format(len(filenames)))

    rec_annotations = []

    rec_annotations_dict = {}
    for s in filenames:
        a = Extract_Annotation_Data(s, root)
        rec_annotations.append(a)
        rec_annotations_dict[s] = a

    no_label_list = []
    crack_list = []
    wheeze_list = []
    both_sym_list = []
    filename_list = []

    for f in filenames:
        d = rec_annotations_dict[f]
        no_labels = len(
            d[(d['Crackles'] == 0) & (d['Wheezes'] == 0)].index)
        n_crackles = len(
            d[(d['Crackles'] == 1) & (d['Wheezes'] == 0)].index)
        n_wheezes = len(
            d[(d['Crackles'] == 0) & (d['Wheezes'] == 1)].index)
        both_sym = len(d[(d['Crackles'] == 1) & (d['Wheezes'] == 1)].index)
        no_label_list.append(no_labels)
        crack_list.append(n_crackles)
        wheeze_list.append(n_wheezes)
        both_sym_list.append(both_sym)
        filename_list.append(f)

    file_label_df = pd.DataFrame(data={'filename': filename_list, 'no label': no_label_list,
                                       'crackles only': crack_list, 'wheezes only': wheeze_list, 'crackles and wheezees': both_sym_list})
    print(file_label_df.head())

    print('Evaluating Imbalance...')
    # print(file_label_df['no label'])
    # evaluate_imbalance(file_label_df['no label'])

    w_labels = file_label_df[(file_label_df['crackles only'] != 0) | (
        file_label_df['wheezes only'] != 0) | (file_label_df['crackles and wheezees'] != 0)]
    # ASSERT LENGTHS?
    # print(file_label_df.sum())
    initial_segments_nb = file_label_df.sum(
    )[1] + file_label_df.sum()[2] + file_label_df.sum()[3] + file_label_df.sum()[4]
    # ASSERT NUMBER
    print("Number of raw audio extracts (ICBHI): {}".format(initial_segments_nb))

    cycle_list = []

    file_id = 0
    for file_name in filenames:
        data = get_icbhi_samples(
            rec_annotations_dict[file_name], file_name, root, sr, file_id, overlap_threshold, audio_length, step_size)
        cycles_with_labels = []
        for d in data[2:]:
            cycles_with_labels.append((d[0], d[1], d[2], file_id))
        cycle_list.extend(cycles_with_labels)
        file_id += 1
        # ASSERT
    print("Number of 2-sec raw audio (ICBHI): {}".format(len(cycle_list)))


    if test:
         # TESTING #1
        print("Running the first set of tests...")
        write_to_file(cycle_list, None, test_files_destination + "/" + "icbhi_cycle_list.pkl")
        write_to_file(rec_annotations_dict, None, test_files_destination + "/" + 
                  "icbhi_rec_annotations_dict.pkl")
        try:
            proc = subprocess.run(["/opt/anaconda3/envs/ObjectDetection/bin/python", str(tests_destination + "/" + "icbhi_tests.py"),
                                "{}".format(audio_length), "{}".format(overlap_threshold), "{}".format(sr), "{}".format(root)], check=True,)  # stdout=PIPE, stderr=PIPE,)
            print("The script passed all the tests.")
        except subprocess.CalledProcessError:
            print("The script failed to pass all the tests. ")
            sys.exit(1)

    # Count of labels across all labels
    no_labels = [c for c in cycle_list if ((c[1] == 0) & (c[2] == 0))]
    c_only = [c for c in cycle_list if ((c[1] == 1) & (c[2] == 0))]
    w_only = [c for c in cycle_list if ((c[1] == 0) & (c[2] == 1))]
    c_w = [c for c in cycle_list if ((c[1] == 1) & (c[2] == 1))]
    
    # split first -> train_dict, test_dict
    train_dict, test_dict = split(
        no_labels, c_only, w_only, c_w, train_test_ratio)
    
    # do augmentation
    if augmentation:
        print("Augmenting...")
        for key in train_dict:
            # train_dict[key] = new_augm_wo_adding(train_dict[key], sr)
            train_dict[key] = new_augm(train_dict[key], sr)
    
    print("Generating spectograms...")
    # convert
    for key in train_dict:
        train_dict[key] = convert_to_spec(train_dict[key], spec_type, sr, n_fft, hop_length, spec_win_length, n_mels)
    for key in test_dict:
        test_dict[key] = convert_to_spec(test_dict[key], spec_type, sr, n_fft, hop_length, spec_win_length, n_mels)
    
    print("Splitting the data...")

    [none_train, c_train, w_train, c_w_train] = [train_dict['none'],
                                                 train_dict['crackles'], train_dict['wheezes'], train_dict['both']]
    [none_test, c_test, w_test, c_w_test] = [test_dict['none'],
                                             test_dict['crackles'], test_dict['wheezes'], test_dict['both']]

    np.random.shuffle(none_train)
    np.random.shuffle(c_train)
    np.random.shuffle(w_train)
    np.random.shuffle(c_w_train)
    print("Done.")


    if test:
        # TESTING #2
        print("Running the second set of tests...")

        write_to_file([none_train, c_train, w_train, c_w_train], None,
                    test_files_destination + "/" + "icbhi_train_data.pkl")
        write_to_file([none_test, c_test, w_test, c_w_test], None, test_files_destination + "/" +
                    "icbhi_val_data.pkl")
        try:
            proc = subprocess.run(["/opt/anaconda3/envs/ObjectDetection/bin/python", tests_destination + "/" + "icbhi_tests_2.py",
                                "{}".format(height), "{}".format(width)], check=True)  # stdout=PIPE, stderr=PIPE,)
        except subprocess.CalledProcessError:
            print("The scripts failed to pass all the tests. ")
            sys.exit(1)

    print('Samples Available')
    print('[Training set]')
    print_sample_count(train_dict)
    print('')
    print('[Test set]')
    print_sample_count(test_dict)

    return [[none_train, c_train, w_train, c_w_train], [none_test, c_test, w_test, c_w_test]]


def generate_perch(root, sr, overlap_threshold, audio_length, step_size, augmentation, train_test_ratio, height, width, spec_type, n_fft, hop_length, spec_win_length, n_mels, tests_destination, test_files_destination, test):
    
    print("Generating PERCH...")
    filenames = [s.split('.')[0] + '.' + s.split('.')[1]
                 for s in os.listdir(path=root) if '.txt' in s]
    # ASSERT
    print("Number of Files: {}".format(len(filenames)))

    rec_annotations = []

    rec_annotations_dict = {}
    for s in filenames:
        a = Extract_Annotation_Data(s, root)
        rec_annotations.append(a)
        rec_annotations_dict[s] = a

    no_label_list = []
    crack_list = []
    wheeze_list = []
    both_sym_list = []
    filename_list = []

    for f in filenames:
        d = rec_annotations_dict[f]
        no_labels = len(
            d[(d['Crackles'] == 0) & (d['Wheezes'] == 0)].index)
        n_crackles = len(
            d[(d['Crackles'] == 1) & (d['Wheezes'] == 0)].index)
        n_wheezes = len(
            d[(d['Crackles'] == 0) & (d['Wheezes'] == 1)].index)
        both_sym = len(d[(d['Crackles'] == 1) & (d['Wheezes'] == 1)].index)
        no_label_list.append(no_labels)
        crack_list.append(n_crackles)
        wheeze_list.append(n_wheezes)
        both_sym_list.append(both_sym)
        filename_list.append(f)

    file_label_df = pd.DataFrame(data={'filename': filename_list, 'no label': no_label_list,
                                       'crackles only': crack_list, 'wheezes only': wheeze_list, 'crackles and wheezees': both_sym_list})
    print(file_label_df.head())

    print('Evaluating Imbalance...')
    evaluate_imbalance(file_label_df['no label'])
 
    w_labels = file_label_df[(file_label_df['crackles only'] != 0) | (
        file_label_df['wheezes only'] != 0) | (file_label_df['crackles and wheezees'] != 0)]
    # ASSERT LENGTHS?
    # print(file_label_df.sum())
    initial_segments_nb = file_label_df.sum(
    )[1] + file_label_df.sum()[2] + file_label_df.sum()[3] + file_label_df.sum()[4]
    # ASSERT NUMBER
    print("Number of raw audio extracts (PERCH): {}".format(initial_segments_nb))

    file_id = 0
    cycle_list = []
    for file_name in filenames:
        data = get_perch_sample(
            rec_annotations_dict[file_name], file_name, root, sr, file_id, audio_length)
        cycles_with_labels = []
        for d in data[2:]:
            cycles_with_labels.append((d[0], d[1], d[2], file_id))
        cycle_list.extend(cycles_with_labels)
        file_id += 1
    print("Number of 2 sec chunks (PERCH): {}".format(len(cycle_list)))

    if test:
        # TESTING #1
        print("Running the first set of tests...")

        write_to_file(cycle_list, None, test_files_destination + "/" + "perch_cycle_list.pkl")
        write_to_file(rec_annotations_dict, None, test_files_destination + "/" +
                    "perch_rec_annotations_dict.pkl")
        try:
            proc = subprocess.run(["/opt/anaconda3/envs/ObjectDetection/bin/python", tests_destination + "/" + "perch_tests.py",
                                "{}".format(audio_length), "{}".format(overlap_threshold), "{}".format(sr), "{}".format(root)], check=True,)  # stdout=PIPE, stderr=PIPE,)
            print("The script passed all the tests.")
        except subprocess.CalledProcessError:
            print("The script failed to pass all the tests. ")
            sys.exit(1)
            
    # Count of labels across all labels
    no_labels = [c for c in cycle_list if ((c[1] == 0) & (c[2] == 0))]
    c_only = [c for c in cycle_list if ((c[1] == 1) & (c[2] == 0))]
    w_only = [c for c in cycle_list if ((c[1] == 0) & (c[2] == 1))]
    c_w = [c for c in cycle_list if ((c[1] == 1) & (c[2] == 1))]


    # split first -> train_dict, test_dict
    train_dict, test_dict = split(
        no_labels, c_only, w_only, c_w, train_test_ratio)
    
    # do augmentation
    if augmentation:
        print("Augmenting...")
        for key in train_dict:
            # train_dict[key] = new_augm_wo_adding(train_dict[key], sr)
            train_dict[key] = new_augm(train_dict[key], sr)
    
    print("Generating spectograms...")
    # convert
    for key in train_dict:
        train_dict[key] = convert_to_spec(train_dict[key], spec_type, sr, n_fft, hop_length, spec_win_length, n_mels)
    for key in test_dict:
        test_dict[key] = convert_to_spec(test_dict[key], spec_type, sr, n_fft, hop_length, spec_win_length, n_mels)
    
    print("Splitting the data...")

    [none_train, c_train, w_train, c_w_train] = [train_dict['none'],
                                                 train_dict['crackles'], train_dict['wheezes'], train_dict['both']]
    [none_test, c_test, w_test, c_w_test] = [test_dict['none'],
                                             test_dict['crackles'], test_dict['wheezes'], test_dict['both']]

    np.random.shuffle(none_train)
    np.random.shuffle(c_train)
    np.random.shuffle(w_train)
    np.random.shuffle(c_w_train)
    print("Done.")


    if test:
        # TESTING #2
        print("Running the second set of tests...")

        write_to_file([none_train, c_train, w_train, c_w_train], None,
                    test_files_destination + "/" + "perch_train_data.pkl")
        write_to_file([none_test, c_test, w_test, c_w_test], None, test_files_destination + "/" +
                    "perch_val_data.pkl")
        try:
            proc = subprocess.run(["/opt/anaconda3/envs/ObjectDetection/bin/python", tests_destination + "perch_tests_2.py",
                                "{}".format(height), "{}".format(width)], check=True)  # stdout=PIPE, stderr=PIPE,)
        except subprocess.CalledProcessError:
            print("The scripts failed to pass all the tests. ")
            sys.exit(1)

    print('Samples Available')
    print('[Training set]')
    print_sample_count(train_dict)
    print('')
    print('[Test set]')
    print_sample_count(test_dict)

    return [[none_train, c_train, w_train, c_w_train], [none_test, c_test, w_test, c_w_test]]


def merge_datasets(dataset_list):
    print("\nMerging Dataset...")
    none_train = []
    c_train = []
    w_train = []
    c_w_train = []
    none_test = []
    c_test = []
    w_test = []
    c_w_test = []

    for dataset in dataset_list:
        none_train += dataset[0][0]
        c_train += dataset[0][1]
        w_train += dataset[0][2]
        c_w_train += dataset[0][3]
        none_test += dataset[1][0]
        c_test += dataset[1][1]
        w_test += dataset[1][2]
        c_w_test += dataset[1][3]

    print('Samples Available')
    print('[Training set]')
    print_sample_count({"none": none_train, "crackles": c_train,
                        "wheezes": w_train, "both": c_w_train})
    print('')
    print('[Test set]')
    print_sample_count({"none": none_test, "crackles": c_test,
                        "wheezes": w_test, "both": c_w_test})
    return [[none_train, c_train, w_train, c_w_train], [none_test, c_test, w_test, c_w_test]]


def new_augm_wo_adding(data, sr):
    print("Inside augmentation w/o adding...")
    min_SNR = 0.001
    max_SNR = 0.5
    ratio = 0.4
    k = int(ratio * len(data)) # how many samples we're augmenting
    
    augment = Compose([
        # AddGaussianNoise(min_amplitude=0.1, max_amplitude=0.15, p=1),
        AddGaussianSNR(min_SNR=min_SNR, max_SNR=max_SNR, p=1),
        # FrequencyMask()
    ])
    
    for i in range(0, k):
        index = random.randint(0, len(data) - 1)
        sample = data[index][0]
        sample = augment(samples=sample, sample_rate=sr)
        whole_tuple = list(data[index])
        whole_tuple[0] = sample
        data[index] = tuple(whole_tuple)
        
    return data
        
def new_augm(data, sr):
    print("Inside new augmentation...")
    # min_SNR=0.01, max_SNR=0.1
    min_SNR = 0.001
    max_SNR = 0.5
    ratio = 0.4
    k = int(ratio * len(data)) # how many samples we're augmenting
    
    portion_to_augment = random.sample(data, k)
    
    augment = Compose([
        # AddGaussianNoise(min_amplitude=0.1, max_amplitude=0.15, p=1),
        AddGaussianSNR(min_SNR=min_SNR, max_SNR=max_SNR, p=1),
        # FrequencyMask()
    ])
    
    #####
    for i in range(len(portion_to_augment)):
        # print(portion_to_augment[i])
        sample = portion_to_augment[i][0]
        # visualize_spec(sample, sr)
        # TODO: ASSIGN NEW ID TO NEW DATA POINT
        sample = augment(samples=sample, sample_rate=sr)
        # visualize_spec(sample, sr)
        whole_tuple = list(portion_to_augment[i])
        whole_tuple[0] = sample
        portion_to_augment[i] = tuple(whole_tuple)
    data += portion_to_augment
    return data

    # sample = portion_to_augment[0][0]
    # # visualize_spec(sample, sr)
    # # TODO: ASSIGN NEW ID TO NEW DATA POINT
    # # print(sample)
    # augmented_sample = augment(samples=sample, sample_rate=sr)
    # # print(augmented_sample)
    # # print(sample == augmented_sample)
    # # print(np.unique(sample == augmented_sample))
    # # visualize_spec(sample, sr)
    # # whole_tuple = list(portion_to_augment[i])
    # # whole_tuple[0] = sample
    # # portion_to_augment[i] = tuple(whole_tuple)
    
    # fs = 8000 #changed by ian 11/6 from a mistake in runme.txt
    # bp = 1
    # cochlear_parameters = {"frmlen": 8*(1/(fs/8000)), "tc": 8, "fac": -2, "shft": np.log2(fs/16000), "FULLT": 0, "FULLX": 0, "bp": bp}
    # coch_path = '/Users/alirachidi/Documents/Sonavi_Labs/classification_algorithm/cochlear_preprocessing/'
    # coch_a = np.loadtxt(coch_path + 'COCH_A.txt', delimiter=',')
    # coch_b = np.loadtxt(coch_path + 'COCH_B.txt', delimiter=',')
    # order = np.loadtxt(coch_path + 'p.txt', delimiter=',')
    # coch_spectrogram = generate_cochlear_spec(augmented_sample, cochlear_parameters, coch_b, coch_a, order)
    # coch_spectrogram = np.transpose(coch_spectrogram)
    # visualize_spec(augmented_sample, sr)
    # exit()

def generate(params):
    print("-----------------------")
    print("Collecting Variables...")
    # Variables
    sr = int(params["SR"])
    icbhi_root = params["ICBHI_ROOT"]
    perch_root = params["PERCH_ROOT"]
    spec_type = params["SPEC_TYPE"]

    # audio_length = Decimal('2')
    # step_size = Decimal('1')
    # overlap_threshold = Decimal('{}'.format(0.15))

    audio_length = Decimal(params["AUDIO_LENGTH"])
    step_size = Decimal(params["STEP_SIZE"])
    overlap_threshold = Decimal(params["OVERLAP_THRESHOLD"])
    # step_size = Decimal('1')
    # overlap_threshold = Decimal('{}'.format(0.15))

    all_file_dest = params["ALL_FILE_DEST"]
    icbhi_file_dest = params["ICBHI_FILE_DEST"]
    perch_file_dest = params["PERCH_FILE_DEST"]

    augmentation = bool(params["AUGMENTATION"])
    test = bool(params["TEST"])
    train_test_ratio = 0.2

    height = int(params["HEIGHT"])
    width = int(params["WIDTH"])

    n_fft = int(params["N_FFT"])
    hop_length = int(params["HOP_LENGTH"])
    spec_win_length = int(params["SPEC_WIN_LENGTH"])
    n_mels = int(params["N_MELS"])

    tests_destination = 'tests/'
    test_files_destination = 'tests/test_files/'

    icbhi_params_list = {
        "root": icbhi_root,
        "sr": sr,
        "overlap_threshold": overlap_threshold,
        "audio_length": audio_length,
        "step_size": step_size,
        "augmentation": augmentation,
        "train_test_ratio": train_test_ratio,
        "height": height,
        "width": width,
        "spec_type": spec_type,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "spec_win_length": spec_win_length,
        "n_mels": n_mels,
        "tests_destination": tests_destination,
        "test_files_destination": test_files_destination,
        "test": test
    }

    perch_params_list = {
        "root": perch_root,
        "sr": sr,
        "overlap_threshold": overlap_threshold,
        "audio_length": audio_length,
        "step_size": step_size,
        "augmentation": augmentation,
        "train_test_ratio": train_test_ratio,
        "height": height,
        "width": width,
        "spec_type": spec_type,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "spec_win_length": spec_win_length,
        "n_mels": n_mels,
        "tests_destination":tests_destination,
        "test_files_destination": test_files_destination,
        "test": test

    }
    
    # write_record(params, icbhi_file_dest)
    # exit()
    
    if bool(params["ALL"]):
        print("List of ICBHI parameters: {}".format(icbhi_params_list))
        icbhi_data = generate_icbhi(**icbhi_params_list)
        print("List of PERCH parameters: {}".format(perch_params_list))
        perch_data = generate_perch(**perch_params_list)
        if bool(params["SAVE_ANY"]) and bool(params["SAVE_IND"]):
            write_to_file(icbhi_data, "ICBHI", icbhi_file_dest)
            write_record(params, icbhi_file_dest)
            write_to_file(perch_data, "PERCH", perch_file_dest)
            write_record(params, perch_file_dest)
        all_data = merge_datasets([icbhi_data, perch_data])
        if bool(params["SAVE_ANY"]):
            write_to_file(all_data, "ALL", all_file_dest)
        write_record(params, all_file_dest)
    else:
        if bool(params["ICBHI"]):
            print("List of ICBHI parameters: {}".format(icbhi_params_list))
            icbhi_data = generate_icbhi(**icbhi_params_list)
            if bool(params["SAVE_ANY"]):
                write_to_file(icbhi_data, "ICBHI", icbhi_file_dest)
                write_record(params, icbhi_file_dest)
        else:
            print("List of PERCH parameters: {}".format(perch_params_list))
            perch_data = generate_perch(**perch_params_list)
            if bool(params["SAVE_ANY"]):
                write_to_file(perch_data, "PERCH", perch_file_dest)
                write_record(params, perch_file_dest)


if __name__ == "__main__":
    seed_everything()
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        "--params",
        help="parameters used in the model and training",
        required=True,
    )
    args = parser.parse_args()
    arguments = args.__dict__
    params = arguments.pop("params")
    params = json.loads(params)
    generate(params)
