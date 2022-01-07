import argparse
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def init():
    print("Collecting Variables...")
    global testing
    global mode

    # most NN parameters
    global n_classes
    global n_epochs
    global lr
    global batch_size
    global ll2_reg
    global weight_decay
    global label_smoothing
    global epsilon
    global shape
    global job_version
    global es_patience
    global min_delta

    # mainly bools for setting parameters in callback, model, etc
    global six
    global concat
    global epoch_start
    global clause
    global target
    global adaptive_lr
    global cuberooting
    global normalizing
    global initial_channels
    global class_weights

    # adaptive lr
    global factor
    global lr_patience
    global min_lr

    # FFT, mel, ASP parameters
    global sr
    global audio_length
    global step_size
    global n_fft
    global hop_length
    global n_mels
    global job_id
    global n_sequences
    global overlap_threshold
    
    # splitting and data handling
    global train_test_ratio
    global kfold

    # paths
    global data_root
    global cache_root
    global file_dir
    global jordan_root
    global icbhi_root
    global bd_root
    global excel_path
    global perch_root
    global ant_root
    global description

    # kapre
    global trainable_fb
    global to_decibel
    
    global train_nn
    

    # cache_root = "/home/alirachidi/classification_algorithm/cache/"
    cache_root = "../cache/"
    # data_root = "/home/alirachidi/classification_algorithm/data/"
    data_root = "../data/"
    jordan_root = os.path.join(data_root, 'jwyy9np4gv-3/')
    icbhi_root = os.path.join(data_root, 'raw_audios/icbhi_preprocessed_v2_cleaned_8000/')
    bd_root = os.path.join(data_root, 'PCV_SEGMENTED_Processed_Files/')
    excel_path = os.path.join(data_root, 'Bangladesh_PCV_onlyStudyPatients.xlsx')
    perch_root = os.path.join(data_root, 'raw_audios/perch_8000_10seconds')
    ant_root = os.path.join(data_root, 'raw_audios/Antwerp_Clinical_Complete')
    # icbhi_root = '../data/raw_audios/icbhi_preprocessed_v2_cleaned_8000/'
    # bd_root = '../data/PCV_SEGMENTED_Processed_Files/'
    # excel_path = "../data/Bangladesh_PCV_onlyStudyPatients.xlsx"
    # perch_root="../data/raw_audios/perch_8000_10seconds/"

    description = None
    job_id = 0
    mode = "pneumonia"
    
    n_classes = 1
    shape = (128, 313)
    n_epochs = 45

    lr = 1e-3
    batch_size = 16
    ll2_reg = 0
    weight_decay = 1e-4
    label_smoothing = 0
    epsilon = 1e-7
    es_patience = 6
    min_delta = 0

    train_test_ratio = 0.8
    kfold = False

    # six = bool(params["SIX"])
    # concat = bool(params["CONCAT"])
    clause = 0
    epoch_start = 0
    testing = 0
    adaptive_lr = 1
    cuberooting = 1
    normalizing = 1
    class_weights = 1
    # oversample =

    initial_channels = 1

    factor = 0.5
    lr_patience = 3
    min_lr = 1e-6

    sr = 8000
    audio_length = 10
    step_size = 5
    n_fft = 1024
    hop_length = 256
    n_mels = 128

    overlap_threshold = 0.15

    # for Kapre
    trainable_fb = False
    to_decibel = True

    train_nn = True

    # old augmentation parameters

    # augmentation = bool(params['AUGMENTATION'])
    # spec_add = bool(spec_params['ADD'])
    # spec_quantity = int(spec_params["QUANTITY"])
    # time_masking = int(spec_params["TIME_MASKING"])
    # frequency_masking = int(spec_params["FREQUENCY_MASKING"])
    # loudness = int(spec_params["LOUDNESS"])

    # jordan_dataset = int(params["JORDAN_DATASET"])
    # pneumonia_only = int(params["PNEUMONIA_ONLY"])

    # wav_add = bool(wav_params['ADD'])
    # wav_quantity = int(wav_params['QUANTITY'])
    # wav_path = str(wav_params["WAV_PATH"]).split(',')
    print("All variables have been collected.")

def return_model_params():
    return { "N_CLASSES": n_classes,
        "SR": sr,
        "BATCH_SIZE": batch_size,
        "LR": lr,
        "SHAPE": shape,
        "INITIAL_CHANNELS": initial_channels,
        "WEIGHT_DECAY": weight_decay,
        "LL2_REG": ll2_reg,
        "EPSILON": epsilon,
        "LABEL_SMOOTHING": label_smoothing
    }

def return_lr_params():
    return {
        "factor": factor,
        "lr_patience": lr_patience,
        "min_lr": min_lr
    }

def parse_arguments():

    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
        "--testing",
        help="activate testing mode or not",
        required=False
    )
    parser.add_argument(
        "--description",
        help="description of the job",
        required=False,
    )
    args = parser.parse_args()
    arguments = args.__dict__
    return arguments

# def set_up_wandb():
    # wandb_name = __file__.split('/')[-1].split('.')[0] + str('_id{}'.format(job_count))
    # print(wandb_name)
    # run = wandb.init(project="tensorboard-integration", name=wandb_name, sync_tensorboard=False)

    # config = wandb.config

    # config.n_classes = n_classes
    # config.n_epochs = n_epochs
    # config.sr = sr
    # config.lr = lr
    # config.batch_size = batch_size
    # config.ll2_reg = ll2_reg
    # config.weight_decay = weight_decay
    # config.label_smoothing = label_smoothing
    # config.es_patience = es_patience
    # config.min_delta = min_delta
    # config.initial_channels = initial_channels
    # config.factor = factor
    # config.lr_patience = lr_patience
    # config.min_lr = min_lr