from modules.helpers import *
from modules.raw_audio import *
from modules.spectrogram import *
from modules.augmentation import *
from modules.testing import *
import warnings

import glob
import statistics
import json
import os
import shutil
import argparse
from decimal import *
import numpy as np
import tensorflow as tf
import librosa

def convert_to_spec(data, spec_type, sr, n_fft, hop_length, spec_win_length, n_mels, coch_path, coch_params):
    
    if spec_type == "log":
        spec_list = convert_to_log(data, n_fft, hop_length, spec_win_length)
    elif spec_type == "coch":
        spec_list = convert_to_coch(data, coch_path, coch_params)
    elif spec_type == "linear":
        spec_list = convert_to_linear(data, n_fft, hop_length)
    else:
        spec_list = convert_to_mel(data, sr, n_fft, hop_length, n_mels)
    return spec_list


def convert_to_linear(data, n_fft, hop_length):
    spec_list = []
    for d in data:
        linear_spectrogram = np.abs(librosa.stft(d[0], n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window='hann')) ** 2
        linear_spectrogram_db = librosa.power_to_db(linear_spectrogram, ref=np.max)
        # visualize_spec(linear_spectrogram_db, 8000, "test")
        # exit()
        spec_list.append([linear_spectrogram_db, d[1], d[2], d[3]])
    return spec_list


def convert_to_log(data, n_fft, hop_length, spec_win_length):
    spec_list = []
    for d in data:
        log_spectrogram = librosa.power_to_db(
            np.abs(librosa.stft(d[0], n_fft=n_fft, hop_length=hop_length, win_length=spec_win_length,
                                center=True, window='hann')) ** 2, ref=1.0)
        spec_list.append([log_spectrogram, d[1], d[2], d[3]])
    return spec_list

def convert_to_coch(data, coch_path, coch_params):
    spec_list = []
    for d in data:
        if (not (len(d[0]) == 80000)):
            continue
        coch_a = np.loadtxt(coch_path + coch_params["COCH_A"], delimiter=',')
        coch_b = np.loadtxt(coch_path + coch_params["COCH_B"], delimiter=',')
        order = np.loadtxt(coch_path + coch_params["P"], delimiter=',')
        coch_spectrogram = generate_cochlear_spec(
            d[0], coch_params, coch_b, coch_a, order)
        coch_spectrogram = np.transpose(coch_spectrogram)
        spec_list.append([coch_spectrogram, d[1], d[2], d[3]])
        if len(spec_list) % 2000 == 0:
            print("Completed {} elements...".format(len(spec_list)))
    return spec_list

def convert_to_mel(data, sr, n_fft, hop_length, n_mels):
    spec_list = []
    for d in data:
        mel_spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(
            d[0], sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels), ref=np.max)
        spec_list.append([mel_spectrogram, d[1], d[2], d[3]])
    return spec_list

def write_bangladesh_to_txt_files(data, txt_dataset_dest):

    f = open(os.path.join(txt_dataset_dest, "aa_paths_and_labels.txt"), 'a')
    for i, spectrogram in enumerate(data):
        file_path = os.path.join(txt_dataset_dest, "{}_{}.txt".format(spectrogram[-1], spectrogram[-2]))
        np.savetxt(file_path, spectrogram[0], delimiter=',')
        f.write('{}, {}\n'.format(file_path, spectrogram[1]))
    f.close()

def convert_pneumonia_label(label):
    if label == "NO PEP":
        return 0
    elif label == "PEP":
        return 1
    else:
        return -1

def find_bangladesh_labels(df, file_name, patient_id):

    # print(file_name)
    # print(patient_id)
    file_column = df.loc[df['HOSP_ID'] == patient_id]
    # print(file_column)
    if file_column.empty:
        return -1
    label = str(file_column['PEP1'].values[0])
    # print(label)
    label = convert_pneumonia_label(label)
    # print(label)
    return label

def get_multiple_bangladesh_samples(file_name, root, sample_rate, df, overlap_threshold=0.15, audio_length=2, step_size=1, extension='.wav'):

    # print(file_name)
    audio_id = str(file_name.split('/')[-1].split('.')[0])
    patient_id = int(file_name.split('/')[-2].split('_')[0])
    # print("Start...")
    # print("Id of the patient: {}".format(patient_id))
    # print("Id of the audio: {}".format(audio_id))
    sample_data = [audio_id]
    data, rate = librosa.load(file_name, sample_rate)
    
    # print("len of the audio: {}".format(len(data)))
    # times, rw_index = find_times(recording_annotations, sample_rate, True)
    rw_index = 0
    max_ind = len(data)
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
                # print("if padded...")
                # print(abs(end_ind - start_ind))
                rw_audio_chunk = generate_padded_samples(
                    rw_audio_chunk, sample_rate * audio_length)
        label = find_bangladesh_labels(df, file_name, patient_id)
        if label == -1:
            break
        sample_data.append([rw_audio_chunk, label])
        rw_index = start + step_size
    return sample_data, len(data)

# def augment_audios(cycles, sr, audio_length, wav_params):
#     for cycle in cycles:
#         # print(cycle)
#         if bool(wav_params['PITCH_SHIFTING']) == True:
#             aug = naa.PitchAug(sampling_rate=sr)
#             cycle[0] = aug.augment(cycle[0])
#         if bool(wav_params['TIME_STRETCHING']) == True:
#             factor = random.uniform(0.6, 1.2)
#             cycle[0] = librosa.effects.time_stretch(cycle[0], factor)
#             # print("start")
#             # print(cycle[0])
#             # print(np.sum(cycle[0]))
#             desired_length = int(sr * audio_length)
#             if len(cycle[0]) >= desired_length:
#                 # print("higher length: {}".format(len(cycle[0])))
#                 cycle[0] = cycle[0][:desired_length]
#             else:
#                 # print("lower length: {}".format(len(cycle[0])))
#                 cycle[0] = generate_padded_samples(cycle[0], desired_length)
#             # print("after")
#             # print(cycle[0])
#             # print(np.sum(cycle[0]))
#             # print(len(cycle[0]))
#         if bool(wav_params['DRC']) == True:
#             pass
#     return cycles


def generate_spectrograms(cycles, spec_type, sr, n_fft, hop_length, spec_win_length, n_mels, coch_path, coch_params, train_test_ratio=0.2):

    print("Generating spectograms...")
    cycles = convert_to_spec(cycles, spec_type, sr, n_fft, hop_length, spec_win_length, n_mels, coch_path, coch_params)
    np.random.shuffle(cycles)
    return cycles

def extract_bangladesh_wav(root, sr, overlap_threshold, length_threshold, audio_length, step_size, dataset=None):

    filenames = []
    y = 0
    for filename in glob.glob(os.path.join(root, "*/*.wav"), recursive=True):
        filenames.append(filename)
    
    print("Number of Files: {}".format(len(filenames)))

    excel_path = "/home/alirachidi/classification_algorithm/data/Bangladesh_PCV_onlyStudyPatients.xlsx"
    df = pd.read_excel(excel_path, engine='openpyxl')

    nb_chunks = 0
    for s in filenames:
        length = librosa.get_duration(filename=s)
        nb_chunks += generate_nb_chunks(length, audio_length, length_threshold, step_size)

    print("Expected number of {}-sec audio chunks: {}".format(audio_length, nb_chunks))

    cycles = []

    for file_name in filenames:
        data, length_2 = get_multiple_bangladesh_samples(file_name, root, sr, df, overlap_threshold, audio_length, step_size) 
        cycles_with_labels = []
        for i, d in enumerate(data[1:]):
            # if i == 0:
            #     print(d[0])
            #     print(np.max(d[0]))
            #     print(np.min(d[0]))
            # print(sum(d[0]))
            # #     # print(np.std(d[0]))
            # plot(d[0], 'original_-1,1_version_{}'.format(i), 'time', 'amplitude')
            # # d[0] = (d[0] - np.mean(d[0])) / np.std(d[0]) # 0-mean, 1-variance norm
            # b=1
            # a=-1
            # audio_max = max(d[0])
            # audio_min = min(d[0])
            # d[0] = [((b-a)*(x-audio_min)/(audio_max-audio_min)+a) for x in d[0]]
            # print(sum(d[0]))
            # #     # audio_sum = sum(d[0])
            # #     # d[0] = [float(i)/audio_sum for i in d[0]]
            # plot(d[0], '-1,1_norm_{}'.format(i), 'time', 'amplitude')
            # exit()
            cycles_with_labels.append([d[0], d[1], i, data[0]])
        if len(cycles) % 500 == 0:
            print("Completed {} elements...".format(len(cycles)))
        cycles.extend(cycles_with_labels)

    print("Number of {}-sec audio chunks: {}".format(audio_length, len(cycles))) 

    return cycles

def generate_bangladesh(bangladesh_root, sr, overlap_threshold, length_threshold, audio_length, step_size, spec_type, n_fft, hop_length, spec_win_length, n_mels, height, width, coch_path, coch_params, wav_params, spec_params, augmentation):

    bangladesh_cycles = extract_bangladesh_wav(bangladesh_root, sr, overlap_threshold, length_threshold, audio_length, step_size, dataset="BANGLADESH")

    if augmentation:
        print("Augmenting the audios..")
        bangladesh_data = augment_audios(bangladesh_cycles, sr, audio_length, wav_params)

    bangladesh_data = generate_spectrograms(bangladesh_cycles, spec_type, sr, n_fft, hop_length, spec_win_length, n_mels, coch_path, coch_params)

    # print("here")
    # batch = []
    # print(len(bangladesh_data))

    # for i in range(len(bangladesh_data)):
    #     # print(bangladesh_data[i])
    #     element = np.expand_dims(bangladesh_data[i][0], axis=-1)
    #     # print(element.shape)
    #     batch.append(element)

    # # print(bangladesh_data.shape)

    # print(batch[0])
    # layer = tf.keras.layers.LayerNormalization()
    # output = layer(batch[0])
    # print(output)
    # # print(output[0])

    # exit()
    
    return bangladesh_data

def generate(params, wav_params, spec_params):
    print("Collecting Variables...")
    # Variables
    
    augmentation = bool(params["AUGMENTATION"])
    
    sr = int(params["SR"])
    # icbhi_root = str(params["ICBHI_ROOT"])
    # perch_root = str(params["PERCH_ROOT"])
    # antwerp_root = str(params["ANTWERP_ROOT"])
    # antwerp_simulated_root = str(params["ANTWERP_SIMULATED_ROOT"])

    spec_type = str(params["SPEC_TYPE"])

    audio_length = Decimal(params["AUDIO_LENGTH"])
    step_size = Decimal(params["STEP_SIZE"])
    overlap_threshold = Decimal(params["OVERLAP_THRESHOLD"])
    length_threshold = Decimal(params["LENGTH_THRESHOLD"])

    all_file_dest = str(params["ALL_FILE_DEST"])

    test = bool(params["TEST"])
    train_test_ratio = 0.2

    height = int(params["HEIGHT"])
    width = int(params["WIDTH"])

    n_fft = int(params["N_FFT"])
    hop_length = int(params["HOP_LENGTH"])
    spec_win_length = int(params["SPEC_WIN_LENGTH"])
    n_mels = int(params["N_MELS"])
    
    fs = int(params["FS"])
    bp = int(params["BP"])
    coch_path = str(params["COCH_PATH"])
    
    bucket_name = "tf_learn_pattern_detection"

    coch_params = {
        "frmlen": 8*(1/(fs/8000)), 
        "tc": 8, 
        "fac": -2, 
        "shft": np.log2(fs/16000), 
        "FULLT": 0, 
        "FULLX": 0, 
        "bp": bp,
        "COCH_A": params["COCH_A"],
        "COCH_B": params["COCH_B"],
        "P": params["P"]
    }

    bangladesh_root = str(params["BANGLADESH_ROOT"])

    bangladesh_params_list = {
        "bangladesh_root" : bangladesh_root, 
        "sr": sr, 
        "overlap_threshold" : overlap_threshold, 
        "length_threshold": length_threshold,
        "audio_length" : audio_length, 
        "step_size" : step_size, 
        "spec_type" : spec_type, 
        "n_fft" : n_fft, 
        "hop_length" : hop_length, 
        "spec_win_length" : spec_win_length, 
        "n_mels" : n_mels, 
        "height" : height, 
        "width" : width, 
        "coch_path" : coch_path, 
        "coch_params" : coch_params,
        "wav_params" : wav_params, 
        "spec_params" : spec_params,
        "augmentation" : augmentation
    }
    
    bangladesh_data = generate_bangladesh(**bangladesh_params_list)

    # Writing dataset
    print("Dataset is completed.")

    folder_name = all_file_dest.split('/')[-1].split('.')[0]

    txt_dataset_dest = '../../data/txt_datasets/{}'.format(folder_name)

    # if not os.path.exists(txt_dataset_dest):
    #     os.mkdir(txt_dataset_dest)
    # else:
    #     shutil.rmtree(txt_dataset_dest)
    if os.path.exists(txt_dataset_dest):
        shutil.rmtree(txt_dataset_dest)
    os.mkdir(txt_dataset_dest)

    write_bangladesh_to_txt_files(bangladesh_data, txt_dataset_dest)

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    seed_everything()
    parser = argparse.ArgumentParser()
    warnings.filterwarnings("ignore", category=UserWarning)
    # Input Arguments
    parser.add_argument(
        "--params",
        help="parameters used in the model and training",
        required=True,
    )
    parser.add_argument(
        "--wav_params",
        help="parameters used for augmenting raw audios",
        required=True,
    )
    parser.add_argument(
        "--spec_params",
        help="parameters used for augmenting spectrograms",
        required=True,
    )
    args = parser.parse_args()
    arguments = args.__dict__
    # print(arguments.pop("params"))
    params = json.loads(arguments.pop("params"))
    wav_params = json.loads(arguments.pop("wav_params"))
    spec_params = json.loads(arguments.pop("spec_params"))
    generate(params, wav_params, spec_params)
