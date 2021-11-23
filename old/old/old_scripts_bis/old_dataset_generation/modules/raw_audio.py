
from decimal import *
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from .helpers import *

def extract_wav(root, sr, overlap_threshold, audio_length, step_size):
    
    extension = '.wav'
    if "PERCH" in root:
        filenames = [s.split('.')[0] + '.' + s.split('.')[1]
                 for s in os.listdir(path=root) if 'F.wav' in s]
    else:
        filenames = [s.split('.')[0]
                    for s in os.listdir(path=root) if extension in s]
    # ASSERT
    print("Number of Files: {}".format(len(filenames)))

    rec_annotations = []

    annotated_dict = {}
    for s in filenames:
        # if "icbhi" in root:
        #     if os.path.exists(os.path.join(root, 'events/' + s + '_events.txt')):
        #         a = Extract_Annotation_Icbhi_Data(s, root)
        #     else:
        #         continue
        # else:
        a = Extract_Annotation_Data(s, root)
        rec_annotations.append(a)
        annotated_dict[s] = a

    # no_label_list = []
    # crack_list = []
    # wheeze_list = []
    # both_sym_list = []
    # filename_list = []

    # for f in filenames:
    #     d = annotated_dict[f]
    #     no_labels = len(
    #         d[(d['Crackles'] == 0) & (d['Wheezes'] == 0)].index)
    #     n_crackles = len(
    #         d[(d['Crackles'] == 1) & (d['Wheezes'] == 0)].index)
    #     n_wheezes = len(
    #         d[(d['Crackles'] == 0) & (d['Wheezes'] == 1)].index)
    #     both_sym = len(d[(d['Crackles'] == 1) & (d['Wheezes'] == 1)].index)
    #     no_label_list.append(no_labels)
    #     crack_list.append(n_crackles)
    #     wheeze_list.append(n_wheezes)
    #     both_sym_list.append(both_sym)
    #     filename_list.append(f)

    # file_label_df = pd.DataFrame(data={'filename': filename_list, 'no label': no_label_list,
    #                                    'crackles only': crack_list, 'wheezes only': wheeze_list, 'crackles and wheezees': both_sym_list})
    # print(file_label_df.head())

    # w_labels = file_label_df[(file_label_df['crackles only'] != 0) | (
    #     file_label_df['wheezes only'] != 0) | (file_label_df['crackles and wheezees'] != 0)]
    # # ASSERT LENGTHS?
    # # print(file_label_df.sum())
    # initial_segments_nb = file_label_df.sum(
    # )[1] + file_label_df.sum()[2] + file_label_df.sum()[3] + file_label_df.sum()[4]
    # ASSERT NUMBER
    # print("Number of raw audio extracts: {}".format(initial_segments_nb))
    cycles = []

    file_id = 0
    for file_name in filenames:
        if file_name == '226_1b1_Pl_sc_LittC2SE':
            continue
        if "icbhi" or "Simulated" in root:
            data = get_icbhi_samples(annotated_dict[file_name], file_name, root, sr, file_id, overlap_threshold, audio_length, step_size, extension) 
            # data = get_icbhi_samples_with_events(annotated_dict[file_name], file_name, root, sr, file_id, overlap_threshold, audio_length, step_size) 
        elif "antwerp" in root:
            data = get_icbhi_samples(annotated_dict[file_name], file_name, root, sr, file_id, overlap_threshold, audio_length, step_size, extension) 
        else:
            data = get_perch_sample(annotated_dict[file_name], file_name, root, sr, file_id, audio_length, extension)
        cycles_with_labels = []
        for d in data[2:]:
            cycles_with_labels.append((d[0], d[1], d[2], file_id))
        cycles.extend(cycles_with_labels)
        file_id += 1

    print("Number of {}-sec raw audio: {}".format(audio_length, len(cycles))) 
    
    cycles = split(cycles) # [[train], [test]]
    return cycles, annotated_dict

def get_icbhi_samples(recording_annotations, file_name, root, sample_rate, file_id, overlap_threshold=0.15, audio_length=2, step_size=1, extension='.wav'):
    sample_data = [file_name, file_id]
    # (rate, data) = read_wav_file(os.path.join(
    #     root, file_name + '.wav'), sample_rate)
    data, rate = librosa.load(os.path.join(
        root, file_name + extension), sample_rate)
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
            if (abs(start_ind - end_ind) < (sample_rate * audio_length * Decimal(0.5))):  # disregard if less than half of the audio length (<1 for 2sec, <5 for 10sec) 
                continue
            else:  # 0 pad if more than half of audio length
                rw_audio_chunk = generate_padded_samples(
                    rw_audio_chunk, sample_rate * audio_length)
        crackles, wheezes = find_labels(
            times, start_ind, end_ind, overlap_threshold, audio_length, sample_rate)
        sample_data.append((rw_audio_chunk, crackles, wheezes))
        rw_index = start + step_size
    return sample_data

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

def get_icbhi_samples_with_events(recording_annotations, file_name, root, sample_rate, file_id, overlap_threshold=0.15, audio_length=2, step_size=1, extension='.wav'):
    sample_data = [file_name, file_id]
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

def get_perch_sample(recording_annotations, file_name, root, sample_rate, file_id, audio_length,  extension='.wav'):
    sample_data = [file_name, file_id]
    # (rate, data) = read_wav_file(os.path.join(
    #     root, file_name + '.wav'), sample_rate)
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

def slice_icbhi_data(start, end, raw_data,  sample_rate, max_ind):
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)
    return raw_data[start_ind: end_ind], start_ind, end_ind


def slice_perch_data(start, end, raw_data, sample_rate):
    start_ind = int(start * sample_rate)
    end_ind = int(end * sample_rate)
    return raw_data[start_ind: end_ind]
