import argparse
import json
import os
import tensorflow as tf
import random
import numpy as np
import shutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def seed_everything():
    os.environ["PYTHONHASHSEED"] = "0"
    np.random.seed(0)
    random.seed(0)
    tf.random.set_seed(0)
    # tf.config.experimental.enable_op_determinism()
    # first = tf.random.normal((10, 1), 1, 1, dtype=tf.float32)
    # print(first)
    # sec = tf.random.normal((10, 1), 1, 1, dtype=tf.float32)
    # print(sec)


def initialize_job():
    global file_dir
    global job_id
    global job_dir
    job_id += 1
    job_dir = os.path.join(file_dir, str(job_id))
    print("First job directory is {}".format(job_dir))
    os.mkdir(job_dir)
    os.mkdir(os.path.join(job_dir, "logs"))
    os.mkdir(os.path.join(job_dir, "logs/train"))
    os.mkdir(os.path.join(job_dir, "logs/validation"))
    os.mkdir(os.path.join(job_dir, "specs"))
    os.mkdir(os.path.join(job_dir, "gradcam"))
    os.mkdir(os.path.join(job_dir, "others"))
    # make more folders
    # initialize the wanddb stuff


def initialize_file_folder(file_dir):
    # maybe delete it too, doing it for now for easy testing
    if os.path.exists(file_dir):
        shutil.rmtree(file_dir)
    os.mkdir(file_dir)
    print("File dir is {}".format(file_dir))


def init(arguments, file_name):
    print("-- Collecting Variables... --")
    print("Tensorflow Version: {}".format(tf.__version__))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
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
    global normalize
    global initial_channels
    global use_class_weights
    global early_stopping
    global oversample

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
    global official_labels_path

    # kapre
    global trainable_fb
    global to_decibel

    global train_nn
    global train_mel
    global spec_aug_params
    global audio_aug_params

    global parse_function

    cache_root = (
        "/Users/alirachidi/Documents/Sonavi_Labs/classification_algorithm/cache/"
    )
    # cache_root = "../../../cache/"  # FIXME posix paths
    data_root = "/Users/alirachidi/Documents/Sonavi_Labs/classification_algorithm/data/"
    # data_root = "../../../data/"
    jordan_root = os.path.join(data_root, "jwyy9np4gv-3/")
    # icbhi_cleaned_root = os.path.join(data_root, 'raw_audios/icbhi_preprocessed_v2_cleaned_8000/')
    icbhi_root = os.path.join(data_root, "raw_audios/icbhi_preprocessed_v2_8000/")
    bd_root = os.path.join(data_root, "PCV_SEGMENTED_Processed_Files/")
    excel_path = os.path.join(data_root, "Bangladesh_PCV_onlyStudyPatients.xlsx")
    perch_root = os.path.join(data_root, "raw_audios/perch_8000_10seconds")
    ant_root = os.path.join(data_root, "raw_audios/Antwerp_Clinical_Complete")
    official_labels_path = os.path.join(
        data_root, "raw_audios/icbhi_preprocessed_v2_8000/official_labels.txt"
    )

    # bd_root = '../data/PCV_SEGMENTED_Processed_Files/'
    # excel_path = "../data/Bangladesh_PCV_onlyStudyPatients.xlsx"
    # perch_root="../data/raw_audios/perch_8000_10seconds/"

    global icbhi_metadata_root
    icbhi_metadata_root = os.path.join(data_root, "raw_audios/demographic_info.txt")

    description = None
    job_id = 0
    mode = "pneumonia"

    # n_classes = 1
    shape = (128, 313)
    n_epochs = 20

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
    normalize = 1
    use_class_weights = 1
    early_stopping = 1
    oversample = 0

    initial_channels = 1

    factor = 0.5
    lr_patience = 3
    min_lr = 1e-6

    sr = 8000
    audio_length = 10
    step_size = (
        5
    )  # jump to the next point, also the overlap between two subsequent chunks frome the same audio

    n_fft = 1024
    hop_length = 256
    n_mels = 128

    # user for labels => how much does the chunk overlap with the times of the label?
    overlap_threshold = 0.15

    # for Kapre
    trainable_fb = False
    to_decibel = True

    train_nn = False
    train_mel = False

    spec_aug_params = []
    audio_aug_params = []

    parse_function = None

    ######
    if arguments["mode"] == "cw":
        mode = "cw"
        class_names = ["none", "crackles", "wheezes", "both"]
    elif arguments["mode"] == "pneumonia":
        mode = "pneumonia"
        class_names = ["negative", "positive"]
    n_classes = 2 if mode == "cw" else 1
    file_dir = os.path.join(cache_root, mode, file_name)
    description = arguments["description"]
    if arguments["testing"]:
        testing = int(arguments["testing"])

    if testing:
        file_dir += "_testing"
        n_epochs = 2
        train_test_ratio = 0.5
        description = "testing"

    # seed_everything()
    initialize_file_folder(file_dir)
    initialize_job()
    print("PID: {}".format(os.getpid()))

    print("Description: {}".format(description))
    print("-- All variables have been collected. --")


def return_model_params():
    return {
        "N_CLASSES": n_classes,
        "SR": sr,
        "BATCH_SIZE": batch_size,
        "LR": lr,
        "SHAPE": shape,
        "INITIAL_CHANNELS": initial_channels,
        "WEIGHT_DECAY": weight_decay,
        "LL2_REG": ll2_reg,
        "EPSILON": epsilon,
        "LABEL_SMOOTHING": label_smoothing,
    }


def return_lr_params():
    return {"factor": factor, "lr_patience": lr_patience, "min_lr": min_lr}


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
        "--testing", help="activate testing mode or not", required=False
    )
    parser.add_argument("--description", help="description of the job", required=False)
    parser.add_argument(
        "--mode",
        help="task at hand, either main for pneumonia or cw for crackles",
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
