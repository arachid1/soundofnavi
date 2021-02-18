
import os
import random
import pickle
import pandas as pd
import numpy as np
import datetime
from google.cloud import storage
import wave
import librosa


def seed_everything():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(0)
    random.seed(0)

def write_record(params, file_path):

    print("Writing record...")
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

def print_sample_count(src_dict):

    print('all:{}\nnone:{}\ncrackles:{}\nwheezes:{}\nboth:{}'.format(
        len(src_dict['none']) + len(src_dict['crackles']) +
        len(src_dict['wheezes']) + len(src_dict['both']),
        len(src_dict['none']),
        len(src_dict['crackles']),
        len(src_dict['wheezes']),
        len(src_dict['both'])))


def Extract_Annotation_Icbhi_Data(file_name, root):

    recording_annotations = pd.read_csv(os.path.join(
        root, 'events/' + file_name + '_events.txt'), names=['Start', 'End', 'Event'], delim_whitespace=True)
    return recording_annotations

def Extract_Annotation_Data(file_name, root):
    recording_annotations = pd.read_csv(os.path.join(
        root, file_name + '.txt'), names=['Start', 'End', 'Crackles', 'Wheezes'], delim_whitespace=True)
    return recording_annotations


def read_wav_file(str_filename, target_rate):
    # wav = wave.open(str_filename, mode='r')
    wav, _ = librosa.load(str_filename)

    (sample_rate, data) = extract2FloatArr(wav, str_filename)

    if (sample_rate != target_rate):
        # print("is resampling...")
        (_, data) = resample(sample_rate, data, target_rate)
    return (target_rate, data.astype(np.float32))

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

def write_to_file(data, name, path):
    print("Writing {} locally...".format(name))
    output = open(path, 'wb')
    pickle.dump(data, output)
    if name is not None:
        print("Find {} dataset at {}".format(name, path))
    output.close()
    
def send_to_gcp(data, name, bucket_name):
    print("Sending to GCP...")
    destination_blob_name = str("datasets/" + name) 
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    string_data = pickle.dumps(data)
    blob.upload_from_string(string_data)
    print("Find dataset at {}".format(destination_blob_name))

def evaluate_imbalance(column):
    neg, pos = np.bincount(column)
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))
