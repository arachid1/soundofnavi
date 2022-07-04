from modules.helpers import *
from modules.raw_audio import *
from modules.pneumonia_icbhi_spectrogram import *
from modules.augmentation import *
from modules.testing import *
import warnings

import json
import os
import argparse
from decimal import *
import numpy as np

def write_pneumonia_icbhi_to_txt_file(data, txt_dataset_dest, dataset_name):

    f = open(os.path.join(txt_dataset_dest, "aa_paths_and_labels.txt"), 'a')
    for i, spectrogram in enumerate(data):
        file_path = os.path.join(txt_dataset_dest, "{}_{}_{}.txt".format(dataset_name, spectrogram[-1], spectrogram[-2]))
        np.savetxt(file_path, spectrogram[0], delimiter=',')
        f.write('{}, {}\n'.format(file_path, spectrogram[1]))
    f.close()


def find_pneumonia_vs_healthy_diagnosis(recording_annotations):
    if recording_annotations['Diagnosis'] == 'Pneumonia':
        return 1
    elif recording_annotations['Diagnosis'] == 'Healthy':
        return 0

def find_pneumonia_diagnosis(recording_annotations):
    if recording_annotations['Diagnosis'] == 'Pneumonia':
        return 1
    else:
        return 0

def find_all_diagnosis(recording_annotations):
    if recording_annotations['Diagnosis'] == 'COPD':
        return 7
    elif recording_annotations['Diagnosis'] == 'Asthma':
        return 6
    elif recording_annotations['Diagnosis'] == 'URTI':
        return 5
    elif recording_annotations['Diagnosis'] == 'Bronchiectasis':
        return 4
    elif recording_annotations['Diagnosis'] == 'LRTI':
        return 3
    elif recording_annotations['Diagnosis'] == 'Bronchiolitis':
        return 2
    elif recording_annotations['Diagnosis'] == 'Pneumonia':
        return 1
    elif recording_annotations['Diagnosis'] == 'Healthy':
        return 0

def get_multiple_icbhi_pneumonia_samples(recording_annotations, file_name, root, sample_rate, overlap_threshold=0.15, audio_length=2, step_size=1, extension='.wav'):
    sample_data = [file_name]
    data, rate = librosa.load(os.path.join(
        root, file_name + extension), sample_rate)
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
                rw_audio_chunk = generate_padded_samples(
                    rw_audio_chunk, sample_rate * audio_length)
        diagnosis = find_pneumonia_diagnosis(recording_annotations)
        # print(diagnosis)
        sample_data.append([rw_audio_chunk, diagnosis])
        rw_index = start + step_size
    return sample_data, len(data)

def generate_icbhi(icbhi_root, sr, overlap_threshold, length_threshold, audio_length, step_size, spec_type, n_fft, hop_length, spec_win_length, n_mels, height, width, coch_path, coch_params, wav_params, spec_params, augmentation):
    print("Generating ICBHI...")
    icbhi_data = extract_pneumonia_icbhi_wav(icbhi_root, sr, overlap_threshold, length_threshold, audio_length, step_size, dataset="ICBHI")
    if augmentation:
        print("Augmenting the audios..")
        icbhi_data = augment_audios(icbhi_data, sr, audio_length, wav_params)

    icbhi_data = generate_spectrograms(icbhi_data, spec_type, sr, n_fft, hop_length, spec_win_length, n_mels, coch_path, coch_params)

    return icbhi_data

def Extract_Annotation_ICBHI_Pneumonia_Data(root):
    recording_annotations = pd.read_csv(os.path.join(
        root, "patient_diagnosis.csv"), names=['Diagnosis'], delimiter=',', index_col=0)
    # print(recording_annotations)
    # print(file_name)
    # patient_id = int(file_name[:3])
    # recording_annotations = recording_annotations.loc[patient_id]
    # print(recording_annotations['Diagnosis'])
    # exit()
    return recording_annotations

def extract_pneumonia_icbhi_wav(root, sr, overlap_threshold, length_threshold, audio_length, step_size, dataset=None, extension='.wav'):
    
    if dataset is None:
        print("Dataset is not specified. Exit.")
        exit()

    if dataset == "PERCH":
        filenames = [s.split('.')[0] + '.' + s.split('.')[1]
                for s in os.listdir(path=root) if s.endswith(extension)]
    else:
        filenames = [s.split('.')[0]
                    for s in os.listdir(path=root) if extension in s]

    filenames = ['207_3b2_Al_mc_AKGC417L', '226_1b1_Pl_sc_LittC2SE']
    # ASSERT
    print("Number of Files: {}".format(len(filenames)))

    rec_annotations = Extract_Annotation_ICBHI_Pneumonia_Data(root)

    print(rec_annotations['Diagnosis'].value_counts())
    
    cycles = []
    for file_name in filenames:
        patient_id = int(file_name[:3])
        # if not ((rec_annotations.loc[patient_id]['Diagnosis'] == 'Healthy') or (rec_annotations.loc[patient_id]['Diagnosis'] == 'Pneumonia')):
        #     continue
        if dataset=="ICBHI" or dataset=="ANTWERP":
            data, length_2 = get_multiple_icbhi_pneumonia_samples(rec_annotations.loc[patient_id], file_name, root, sr, overlap_threshold, audio_length, step_size, extension) 
        else:
            data = get_single_sample(rec_annotations.loc[patient_id], file_name, root, sr, audio_length, extension)
        cycles_with_labels = []
        for i, d in enumerate(data[1:]):
            cycles_with_labels.append([d[0], d[1], i, file_name])
        cycles.extend(cycles_with_labels)
    # print("Expected number of {}-sec audio chunks BIS: {}".format(audio_length, nb_chunks_2))

    print("Number of {}-sec audio chunks: {}".format(audio_length, len(cycles))) 

    return cycles

def generate(params, wav_params, spec_params):
    print("Collecting Variables...")
    # Variables
    
    augmentation = bool(params["AUGMENTATION"])
    
    sr = int(params["SR"])
    icbhi_root = str(params["ICBHI_ROOT"])
    perch_root = str(params["PERCH_ROOT"])
    antwerp_root = str(params["ANTWERP_ROOT"])
    antwerp_simulated_root = str(params["ANTWERP_SIMULATED_ROOT"])

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

    icbhi_params_list = {
        "icbhi_root" : icbhi_root, 
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
    
    perch_params_list = {
        "perch_root" : perch_root, 
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

    antwerp_params_list = {
        "antwerp_root" : antwerp_root, 
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

    antwerp_simulated_params_list = {
        "antwerp_root" : antwerp_simulated_root, 
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

    # antwerp_data = generate_antwerp(**antwerp_params_list)
    # perch_data = generate_perch(**perch_params_list)
    # antwerp_simulated_data = generate_antwerp(**antwerp_simulated_params_list)
    icbhi_data = generate_icbhi(**icbhi_params_list)
    
    print("All datasets are completed.")

    # Writing dataset
    folder_name = all_file_dest.split('/')[-1].split('.')[0]

    txt_dataset_dest = '../../data/txt_datasets/{}'.format(folder_name)

    if not os.path.exists(txt_dataset_dest):
        os.mkdir(txt_dataset_dest)

    print("Writing...")

    # write_to_txt_files(antwerp_data, txt_dataset_dest, "antwerp")
    # write_to_txt_files(antwerp_simulated_data, txt_dataset_dest, "antwerp_sim")
    write_pneumonia_icbhi_to_txt_file(icbhi_data, txt_dataset_dest, "icbhi")
    # write_to_txt_files(perch_data, txt_dataset_dest, "perch")

    print("Find dataset at {}".format(txt_dataset_dest))



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
