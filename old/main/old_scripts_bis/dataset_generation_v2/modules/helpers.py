from matplotlib import pyplot as plt
import os
import random
import pickle
import pandas as pd
import numpy as np
import datetime
from google.cloud import storage
import wave
import librosa
from librosa import display

def seed_everything():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(0)
    random.seed(0)

def generate_nb_chunks(length, audio_length, length_threshold, step_size):
    # print(length)
    # print(step_size)
    nb_chunks = 0
    while length > float(step_size): # >= for consistency with get_sample functions
        length -= float(step_size)
        nb_chunks += 1
    # print(nb_chunks)
    return nb_chunks

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

def visualize_spec(data, sr, name):
    
    fig = plt.figure(figsize=(20, 10))
    display.specshow(
        data,
        # y_axis="log",
        sr=sr,
        cmap="coolwarm"
    )
    plt.colorbar()
    plt.show()
    plt.savefig("{}.png".format(name))

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

def write_to_txt_files(data, txt_dataset_dest, dataset_name):

    f = open(os.path.join(txt_dataset_dest, "aa_paths_and_labels.txt"), 'a')
    for i, spectrogram in enumerate(data):
        file_path = os.path.join(txt_dataset_dest, "{}_{}_{}.txt".format(dataset_name, spectrogram[-1], spectrogram[-2]))
        np.savetxt(file_path, spectrogram[0], delimiter=',')
        f.write('{}, {}, {}\n'.format(file_path, spectrogram[1], spectrogram[2]))
    f.close()


def write_to_pickle_file(data, name, path):
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

def plot(vector, name, xlabel=None, ylabel=None):
    plt.figure()
    plt.plot(vector)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot()
    plt.savefig(name)