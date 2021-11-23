
from decimal import *
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import wave
from .helpers import *

def extract_wav(root, sr, overlap_threshold, length_threshold, audio_length, step_size, dataset=None, extension='.wav'):
    
    if dataset is None:
        print("Dataset is not specified. Exit.")
        exit()

    if dataset == "PERCH":
        filenames = [s.split('.')[0] + '.' + s.split('.')[1]
                for s in os.listdir(path=root) if s.endswith(extension)]
    else:
        filenames = [s.split('.')[0]
                    for s in os.listdir(path=root) if extension in s]

    # ASSERT
    # filenames = filenames[:10]
    print("Number of Files: {}".format(len(filenames)))

    rec_annotations = []

    annotated_dict = {}
    nb_chunks = 0
    for s in filenames:
        a = Extract_Annotation_Data(s, root)
        # length = float(a['End'][len(a) - 1])
        # print(a['End'])
        length = librosa.get_duration(filename=os.path.join(root, s + '.wav'))
        # print(a['End'][len(a) - 1])
        # print("here")
        # print(length)
        nb_chunks += generate_nb_chunks(length, audio_length, length_threshold, step_size)
        # print(nb_chunks)
        rec_annotations.append(a)
        annotated_dict[s] = a
        # if s == filenames[100]:
        #     exit()
    print("Expected number of {}-sec audio chunks: {}".format(audio_length, nb_chunks))
    
    cycles = []
    for file_name in filenames:
        if dataset=="ICBHI" or dataset=="ANTWERP":
            data, length_2 = get_multiple_samples(annotated_dict[file_name], file_name, root, sr, overlap_threshold, audio_length, step_size, extension) 
        else:
            data = get_single_sample(annotated_dict[file_name], file_name, root, sr, audio_length, extension)
        # nb_chunks_2 += generate_nb_chunks(length_2, audio_length, length_threshold, step_size)
        cycles_with_labels = []
        for i, d in enumerate(data[1:]):
            cycles_with_labels.append((d[0], d[1], d[2], i, file_name))
        cycles.extend(cycles_with_labels)
    # print("Expected number of {}-sec audio chunks BIS: {}".format(audio_length, nb_chunks_2))

    print("Number of {}-sec audio chunks: {}".format(audio_length, len(cycles))) 

    return cycles, annotated_dict

def get_multiple_samples(recording_annotations, file_name, root, sample_rate, overlap_threshold=0.15, audio_length=2, step_size=1, extension='.wav'):
    sample_data = [file_name]
    data, rate = librosa.load(os.path.join(
        root, file_name + extension), sample_rate)
    # print("len: {}".format(len(data)))
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
            # print(abs(start_ind - end_ind))
            if (abs(start_ind - end_ind) < (sample_rate * audio_length * Decimal(0.5))):  # disregard if less than half of the audio length (<1 for 2sec, <5 for 10sec) 
                continue
            else:  # 0 pad if more than half of audio length
                # print("here")
                rw_audio_chunk = generate_padded_samples(
                    rw_audio_chunk, sample_rate * audio_length)
        crackles, wheezes = find_labels(
            times, start_ind, end_ind, overlap_threshold, audio_length, sample_rate)
        sample_data.append((rw_audio_chunk, crackles, wheezes))
        rw_index = start + step_size
    # print(len(data))
    return sample_data, len(data)

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

def process_overlaps(overlaps):
    crackles = sum([overlap[2] for overlap in overlaps])
    wheezes = sum([overlap[3] for overlap in overlaps])
    crackles = 1.0 if (not(crackles == 0)) else 0.0
    wheezes = 1.0 if (not(wheezes == 0)) else 0.0
    return crackles, wheezes

def get_multiple_samples_with_events(recording_annotations, file_name, root, sample_rate, overlap_threshold=0.15, audio_length=2, step_size=1, extension='.wav'):
    sample_data = [file_name, ]
    # (rate, data) = read_wav_file(os.path.join(
    #     root, file_name + '.wav'), sample_rate)
    data, rate = librosa.load(os.path.join(
        root, file_name + extension), sample_rate)
    times = find_times_with_events(recording_annotations, sample_rate) ######
    rw_index = 0
    max_ind = len(data)
    # max_ind = int(times[len(times) - 1][1])
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
            if (abs(start_ind - end_ind) < (sample_rate * audio_length * Decimal(0.5))):  # disregard if less than half of the audio length (<1 for 2sec, <5 for 10sec) 
                continue
            else:  # 0 pad if more than half of audio length
                rw_audio_chunk = generate_padded_samples(
                    rw_audio_chunk, sample_rate * audio_length)
        if len(times) == 0:
            crackles = 0
            wheezes = 0
        else: 
            crackles, wheezes = find_labels_with_events(
                times, start_ind, end_ind, overlap_threshold, audio_length, sample_rate)  #####
        sample_data.append((rw_audio_chunk, crackles, wheezes))
        rw_index = start + step_size
    return sample_data


def find_times_with_events(recording_annotations, sample_rate):
    times = []
    if not recording_annotations.empty:
        for i in range(len(recording_annotations.index)):
            row = recording_annotations.loc[i]
            c_start = Decimal('{}'.format(row['Start']))
            c_end = Decimal('{}'.format(row['End']))
            if row['Event'] == "crackle":
                times.append((c_start * sample_rate, c_end * sample_rate,
                            1, 0)) 
            else:
                times.append((c_start * sample_rate, c_end * sample_rate,
                            0, 1)) 
    return times

def find_labels_with_events(times, start, end, overlap_threshold, audio_length, sample_rate):
    overlaps = []
    for time in times:
        # if (start > time[0] or end < time[1]:
        if start > time[0]:  # the window starts after
            if start >= time[1]:  # no overlap
                continue
            if end <= time[1]:  # perfect overlap: window inside breathing pattern
                return time[2], time[3]
            else:  # time[1] < end: partial overlap, look at next window c)
                overlaps.append(time)
        else:  # the window starts before
            if end <= time[0]:  # no overlap
                continue
            if end < time[1]:  # partial overlap b)
                overlaps.append(time)
            else:  # end > time[1]: perfect overlap: breathing pattern inside of window
                overlaps.append(time)
    return process_overlaps(overlaps)

def get_single_sample(recording_annotations, file_name, root, sample_rate, audio_length,  extension='.wav'):
    sample_data = [file_name, ]
    data, rate = librosa.load(os.path.join(
        root, file_name + extension), sample_rate)
    row = recording_annotations.loc[0]
    c_start = Decimal('{}'.format(row['Start']))
    c_end = Decimal('{}'.format(row['End']))
    rw_audio_chunk = slice_perch_data(c_start, c_end, data, rate)
    if len(rw_audio_chunk) < sample_rate * audio_length:
        rw_audio_chunk = generate_padded_samples(
            rw_audio_chunk, sample_rate * audio_length)
    sample_data.append((rw_audio_chunk, row['Crackles'], row['Wheezes']))
    return sample_data

def split(cycles, train_test_ratio=0.2):
    
    no_labels = [c for c in cycles if ((c[1] == 0) & (c[2] == 0))]
    c_only = [c for c in cycles if ((c[1] == 1) & (c[2] == 0))]
    w_only = [c for c in cycles if ((c[1] == 0) & (c[2] == 1))]
    c_w = [c for c in cycles if ((c[1] == 1) & (c[2] == 1))]

    if len(no_labels) != 0: #TODO: update this quick fix
        none_train, none_test = train_test_split(
            no_labels, test_size=train_test_ratio)
    else:
        none_train = []
        none_test = []
    c_train, c_test = train_test_split(c_only, test_size=train_test_ratio)
    w_train, w_test = train_test_split(w_only, test_size=train_test_ratio)
    c_w_train, c_w_test = train_test_split(c_w, test_size=train_test_ratio)

    return [[none_train, c_train, w_train, c_w_train], [none_test, c_test, w_test, c_w_test]]

def slice_icbhi_data(start, end, raw_data, sample_rate, max_ind):
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)
    return raw_data[start_ind: end_ind], start_ind, end_ind


def slice_perch_data(start, end, raw_data, sample_rate):
    start_ind = int(start * sample_rate)
    end_ind = int(end * sample_rate)
    return raw_data[start_ind: end_ind]

# def Extract_Annotation_Icbhi_Events_Data(file_name, root):
#     recording_annotations = pd.read_csv(os.path.join(
#         root, 'events/' + file_name + '_events.txt'), names=['Start', 'End', 'Event'], delim_whitespace=True)
#     return recording_annotations

def Extract_Annotation_Data(file_name, root):
    recording_annotations = pd.read_csv(os.path.join(
        root, file_name + '.txt'), names=['Start', 'End', 'Crackles', 'Wheezes'], delim_whitespace=True)
    return recording_annotations