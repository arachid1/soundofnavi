# from .modules.helpers import *
# from .modules.generators import *
# from .modules.metrics import *
# from .modules.callbacks import *
# from .modules.pneumonia import *
# from .modules.parse_functions import *
# from .modules.augmentation import *
# from .core import *
from .modules.main import parameters
from .modules.main.helpers import *

from .modules.audio_loader.JordanAudioLoader import JordanAudioLoader
from .modules.audio_loader.IcbhiAudioLoader import IcbhiAudioLoader
from .modules.audio_loader.helpers import default_get_filenames

# from .modules.audio_preparer.helpers import *

from .modules.spec_generator.SpecGenerator import SpecGenerator

# from .modules.augmenter.helpers import *

from .modules.callbacks.MetricsCallback import MetricsCallback

from .models import mixednet

from .modules.parse_functions.parse_functions import generate_spec

from .modules.augmenter.functions import stretch_time, shift_pitch 

# from .modules.main.global_helpers import visualize_spec, pad_sample

from sklearn.utils import class_weight
import tensorflow as tf

# from .models import *

def train_model(model_to_be_trained, spec_aug_params, audio_aug_params):
    # logs_path = job_dir + "/logs/"
    jordan_root = '../../data/jwyy9np4gv-3/'
    icbhi_root = '../../data/raw_audios/icbhi_preprocessed_v2_cleaned_8000/'

    # full audios
    IcbhiAudioLoaderObj = IcbhiAudioLoader(icbhi_root, default_get_filenames)
    JordanAudioLoaderObj = JordanAudioLoader(jordan_root, default_get_filenames)
    audio_loaders = [IcbhiAudioLoaderObj, JordanAudioLoaderObj]
    audios_dict = load_audios(audio_loaders)

    # audio chunks
    audios_c_dict = prepare_audios(audios_dict)

    # split and extend
    train_audios_c_dict, val_audios_c_dict = split_and_extend(audios_c_dict, parameters.train_test_ratio)

    # val
    val_samples = generate_samples(val_audios_c_dict)

    # train
    train_samples, original_training_length = set_up_training_samples(train_audios_c_dict, spec_aug_params, audio_aug_params)

    # tf datasets
    train_dataset, __, train_labels, __ = create_tf_dataset(train_samples, generate_spec, batch_size=parameters.batch_size, shuffle=True)
    val_dataset, val_specs, val_labels, val_filenames = create_tf_dataset(val_samples, generate_spec, batch_size=1, shuffle=False)
    train_non_pneumonia_nb, train_pneumonia_nb = train_labels.count(0), train_labels.count(1)
    print("--- Final training dataset went from {} to {} elements, with {} 0's, {} 1's and {} others ---".format(original_training_length, len(train_samples), train_non_pneumonia_nb, train_pneumonia_nb, len(train_labels) - train_non_pneumonia_nb - train_pneumonia_nb))
    print("--- Validation dataset contains {} elements, with {} 0's and {} 1's ---".format(len(val_samples), val_labels.count(0), val_labels.count(1)))

    # weights
    weights = None

    if bool(parameters.class_weights):
        print("Initializing weights...")
        weights = class_weight.compute_class_weight(
            "balanced", [0, 1], [l for l in train_labels if l == 0 or l == 1])
        weights = {i: weights[i] for i in range(0, len(weights))}
        print("weights = {}".format(weights))
    
    # callbacks
    metrics_callback = MetricsCallback(val_dataset, val_filenames, parameters.shape, parameters.initial_channels, parameters.n_classes, parameters.job_dir, parameters.sr, parameters.min_delta, parameters.es_patience, parameters.patience, parameters.min_lr, parameters.factor, parameters.epoch_start, parameters.target, parameters.job_id, parameters.clause, None, None, parameters.cuberooting, parameters.normalizing, adaptive_lr=parameters.adaptive_lr)

    gpus = tf.config.experimental.list_logical_devices('GPU')

    # model setting
    model = model_to_be_trained(**parameters.return_model_params())

    model.summary(line_length=110)

    if len(gpus) > 1:
        print("You are using 2 GPUs while the code is set up for one only.")
        exit()

    # training
    model.fit(
        train_dataset,
        epochs=parameters.n_epochs,
        verbose=2,
        class_weight=weights,
        callbacks=[metrics_callback]
    )

    model.save(parameters.job_dir + "/{}/model_{}.h5".format(parameters.job_id, parameters.n_epochs))

if __name__ == "__main__":
    # print("Tensorflow Version: {}".format(tf.__version__))
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    seed_everything()
    _, params, _, _ = parameters.parse_arguments()
    parameters.init(params)
    parameters.train_test_ratio = 0.7
    initialize_folder()
    print("-----------------------")
    initialize_job()
    spec_aug_params = [["mixup", {"quantity" : 1, "no_pad" : True, "label_one" : 0, "label_two" : 1, "minval" : 0.3, "maxval" : 0.7}],
       ["cutmix", {"quantity" : 1, "no_pad" : True, "label_one" : 0, "label_two" : 1, "minval" : 0.3, "maxval" : 0.7}]]
    audio_aug_params = [["augmix", {"quantity" : 0.4, "label": -1, "no_pad" : False, "minval" : 0.3, "maxval" : 0.7, "aug_functions": [shift_pitch, stretch_time]}]]
    train_model(mixednet, spec_aug_params, audio_aug_params)
    initialize_job()
    spec_aug_params = [["mixup", {"quantity" : 2, "no_pad" : True, "label_one" : 0, "label_two" : 1, "minval" : 0.3, "maxval" : 0.7}],
       ["cutmix", {"quantity" : 2, "no_pad" : True, "label_one" : 0, "label_two" : 1, "minval" : 0.3, "maxval" : 0.7}]]
    train_model(mixednet, spec_aug_params, audio_aug_params)
    initialize_job()
    spec_aug_params = [["mixup", {"quantity" : 2, "no_pad" : True, "label_one" : 0, "label_two" : 1, "minval" : 0.3, "maxval" : 0.7}],
       ["cutmix", {"quantity" : 2, "no_pad" : True, "label_one" : 0, "label_two" : 1, "minval" : 0.3, "maxval" : 0.7}]]
    train_model(mixednet, spec_aug_params, audio_aug_params)
    # initialize_job()
    # audio_aug_params = [["augmix", {"quantity" : 0.4, "label": -1, "no_pad" : False, "minval" : 0.3, "maxval" : 0.7, "aug_functions": [stretch_time]}]]
    # train_model(mixednet, spec_aug_params, audio_aug_params)

