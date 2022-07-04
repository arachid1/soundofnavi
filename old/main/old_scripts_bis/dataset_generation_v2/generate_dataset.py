from modules.helpers import *
from modules.raw_audio import *
from modules.spectrogram import *
from modules.augmentation import *
from modules.testing import *
import warnings

import json
import os
import argparse
from decimal import *
import numpy as np

def generate_icbhi(icbhi_root, sr, overlap_threshold, length_threshold, audio_length, step_size, spec_type, n_fft, hop_length, spec_win_length, n_mels, height, width, coch_path, coch_params, wav_params, spec_params, augmentation):
    print("Generating ICBHI...")
    icbhi_cycles, icbhi_dict = extract_wav(icbhi_root, sr, overlap_threshold, length_threshold, audio_length, step_size, dataset="ICBHI")
    # print(len(icbhi_cycles))
    icbhi_cycles = augment_wav(icbhi_cycles, sr, augmentation, **wav_params)
    # print(len(icbhi_cycles))
    # test_wav(icbhi_cycles, icbhi_dict, icbhi_root, sr, audio_length, overlap_threshold, "icbhi")

    icbhi_data = generate_spectrograms(icbhi_cycles, spec_type, sr, n_fft, hop_length, spec_win_length, n_mels, coch_path, coch_params)
    # print(len(icbhi_data))    
    icbhi_data = augment_spectrogram(icbhi_data, sr, augmentation, **spec_params)
    # test_spectrograms(icbhi_data, height, width, "icbhi")
    # print(len(icbhi_data))

    return icbhi_data

def generate_perch(perch_root, sr, overlap_threshold, length_threshold, audio_length, step_size, spec_type, n_fft, hop_length, spec_win_length, n_mels, height, width, coch_path, coch_params, wav_params, spec_params, augmentation):
    print("Generating PERCH...")
    
    perch_cycles, perch_dict = extract_wav(perch_root, sr, overlap_threshold, length_threshold, audio_length, step_size, dataset="PERCH")
    perch_cycles = augment_wav(perch_cycles, sr, augmentation, **wav_params)
    # test_wav(perch_cycles, perch_dict, perch_root, sr, audio_length, overlap_threshold, "perch")

    perch_data = generate_spectrograms(perch_cycles, spec_type, sr, n_fft, hop_length, spec_win_length, n_mels, coch_path, coch_params)
    perch_data = augment_spectrogram(perch_data, sr, augmentation, **spec_params)
    # test_spectrograms(perch_data, height, width, "perch")
    return perch_data


def generate_antwerp(antwerp_root, sr, overlap_threshold, length_threshold, audio_length, step_size, spec_type, n_fft, hop_length, spec_win_length, n_mels, height, width, coch_path, coch_params, wav_params, spec_params, augmentation):
    print("Generating ANTWERP...")
    
    antwerp_cycles, antwerp_dict = extract_wav(antwerp_root, sr, overlap_threshold, length_threshold, audio_length, step_size, dataset="ANTWERP")
    antwerp_cycles = augment_wav(antwerp_cycles, sr, augmentation, **wav_params)
    # test_wav(antwerp_cycles, antwerp_dict, antwerp_root, sr, audio_length, overlap_threshold, "antwerp")

    antwerp_data = generate_spectrograms(antwerp_cycles, spec_type, sr, n_fft, hop_length, spec_win_length, n_mels, coch_path, coch_params)
    antwerp_data = augment_spectrogram(antwerp_data, sr, augmentation, **spec_params)
    # test_spectrograms(antwerp_data, height, width, "antwerp")
    return antwerp_data

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

    antwerp_data = generate_antwerp(**antwerp_params_list)
    perch_data = generate_perch(**perch_params_list)
    antwerp_simulated_data = generate_antwerp(**antwerp_simulated_params_list)
    icbhi_data = generate_icbhi(**icbhi_params_list)
    
    print("All datasets are completed.")

    # Writing dataset
    folder_name = all_file_dest.split('/')[-1].split('.')[0]

    txt_dataset_dest = '../../data/txt_datasets/{}'.format(folder_name)

    if not os.path.exists(txt_dataset_dest):
        os.mkdir(txt_dataset_dest)

    print("Writing...")

    write_to_txt_files(antwerp_data, txt_dataset_dest, "antwerp")
    write_to_txt_files(antwerp_simulated_data, txt_dataset_dest, "antwerp_sim")
    write_to_txt_files(icbhi_data, txt_dataset_dest, "icbhi")
    write_to_txt_files(perch_data, txt_dataset_dest, "perch")

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
