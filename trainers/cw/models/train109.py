import sys
# sys.path.insert(0, 'main/models/conv')
from modules.main import parameters
from modules.main.helpers import *
from modules.main.global_helpers import *

from modules.audio_loader.JordanAudioLoader import JordanAudioLoader
from modules.audio_loader.IcbhiAudioLoader import IcbhiAudioLoader
from modules.audio_loader.BdAudioLoader import BdAudioLoader
from modules.audio_loader.PerchAudioLoader import PerchAudioLoader
from modules.audio_loader.helpers import default_get_filenames, bd_get_filenames, perch_get_filenames

from modules.spec_generator.SpecGenerator import SpecGenerator

from modules.callbacks.NewCallback import NewCallback
from modules.callbacks.visualizationCallback import visualizationCallback

from modules.models.leaf_model9_model_106 import leaf_model9_model_106

from modules.parse_functions.parse_functions import spec_parser, generate_timed_spec

from modules.augmenter.functions import stretch_time, shift_pitch 
from modules.models.core import Distiller

from collections import Counter
from sklearn.utils import class_weight
import tensorflow as tf
import tensorflow_addons as tfa
import leaf_audio.frontend as leaf_frontend
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Optional

# class leaf_resnet_bis(tf.keras.Model):
#   """Neural network architecture to train an audio classifier from waveforms."""

#   def __init__(self,
#                num_outputs: int,
#                _frontend: Optional[tf.keras.Model] = None,
#                encoder: Optional[tf.keras.Model] = None,
#                normalize: bool = True):
#     """Initialization.
#     Args:
#       num_outputs: the number of classes of the classification problem.
#       frontend: A keras model that takes a waveform and outputs a time-frequency
#         representation.
#       encoder: An encoder to turn the time-frequency representation into an
#         embedding.
#     """
#     super().__init__()
#     self._frontend = _frontend
#     self._mel_frontend = leaf_frontend.MelFilterbanks(sample_rate=self._frontend.sample_rate, n_filters=self._frontend.n_filters)
#     self._sinc_frontend = leaf_frontend.SincNet(sample_rate=self._frontend.sample_rate, n_filters=self._frontend.n_filters)
#     self._leaf_frontend_bis = leaf_frontend.Leaf(sample_rate=8000, n_filters=80, window_stride=20)
#     self._encoder = encoder
#     self.normalize = normalize
#     KERNEL_SIZE = 6
#     POOL_SIZE = (2, 2)
#     self.bce = tf.keras.losses.BinaryCrossentropy()
#     self._pool =  tf.keras.Sequential([
#         tf.keras.applications.EfficientNetB3(
#         include_top=False,
#         weights=None,
#         input_tensor=None,
#         input_shape=(80,250,3),
#         pooling=None,
#         classes=num_outputs,
#         ),
#         tf.keras.layers.GlobalAveragePooling2D(),
#         tf.keras.layers.Dense(64, activation="relu"),
#         tf.keras.layers.Dropout(0.5, ),
#         tf.keras.layers.Dense(num_outputs, activity_regularizer=tf.keras.regularizers.l2(parameters.ll2_reg), activation="sigmoid")
        
#     ])


#   def compute_mr_spectral_loss(self, x):
#       spec = self(x, return_spec=True)
#       mel_spec = self._mel_frontend(x)
#       sinc_spec = self._sinc_frontend(x)
#       leaf_bis = self._leaf_frontend_bis(x)
#       if self.normalize:
#           mel_spec = mel_spec - tf.reduce_min(mel_spec)
#           mel_spec = mel_spec / tf.reduce_max(mel_spec)
#           mel_spec = mel_spec*2-1
#           sinc_spec = sinc_spec - tf.reduce_min(sinc_spec)
#           sinc_spec = sinc_spec / tf.reduce_max(sinc_spec)
#           sinc_spec = sinc_spec*2-1
#           leaf_bis = leaf_bis - tf.reduce_min(leaf_bis)
#           leaf_bis = leaf_bis / tf.reduce_max(leaf_bis)
#           leaf_bis = leaf_bis*2-1

#       loss = tf.reduce_mean(tf.math.square(mel_spec-spec))
#       # tf.print(loss)
#       loss += tf.reduce_mean(tf.math.square(sinc_spec-spec))
#       loss += tf.reduce_mean(tf.math.square(leaf_bis-spec))
#       # tf.print(loss)
#       loss = tf.math.sqrt(loss)
#       # tf.print(loss)

#       return loss

#   def compute_loss(self, x, y, y_pred, sample_weight):
#       loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
#       # tf.print("first")
#       # tf.print(loss)
#       # loss += self.compute_mr_spectral_loss(x)
#     #   tf.print(loss)
#       return loss

#   def call(self, inputs: tf.Tensor, training: bool = True, return_spec: bool = False):
#     output = inputs
#     if self._frontend is not None:
#       output = self._frontend(output, training=training)  # pylint: disable=not-callable
#       if self.normalize:
#           output = output/tf.reduce_max(output)
#           output = 2 * output - 1
#       if return_spec:
#           return output
#       output = tf.expand_dims(output, -1)
#     if self._encoder:
#       output = self._encoder(output, training=training)
#     output = tf.transpose(output, [0, 2, 1, 3])
#     # tf.print(tf.shape(output))
#     output = tf.repeat(output, repeats=[3], axis=3)
#     # b = tf.tile(a, multiples)
#     # tf.print(tf.shape(output))

#     output = self._pool(output)
#     # tf.print(tf.shape(output))
#     # output = self._dense(output)
#     # tf.print(tf.shape(output))
#     return(output)


#   def save(self, dest, epoch):
#     self._frontend.save_weights(dest + "/_frontend_{}.h5".format(epoch))
#     self._pool.save_weights(dest + "/_pool{}.h5".format(epoch))


#   def _load(self, source, epoch):
#     self._frontend.load_weights(source + "_frontend_{}.h5".format(epoch))
#     self._pool.load_weights(source + "_pool{}.h5".format(epoch))



class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, learning_rate, _id):
    self.initial_learning_rate = learning_rate
    self.identifier = _id
    self.count = 2

  # @tf.function
  def __call__(self, step):
    # n_steps_per_epochs = int(2492 / parameters.batch_size)

    # if step == 0 or step ==n_steps_per_epochs or step == n_steps_per_epochs*15 or step == n_steps_per_epochs*30 or step == n_steps_per_epochs*45:
    # tf.print("New learning rate for {} ".format(self.identifier))
    # tf.print(self.initial_learning_rate / tf.math.ceil((step+1)/ (2492 / parameters.batch_size)))
    #    tf.print(self.lr)
    #    self.lr = tf.math.divide(self.lr, self.count)
    #    tf.print(self.lr)
    #    tf.print(self.count)
    #    self.count == self.count + 1
    #    tf.print(self.count)

    # return self.lr  learning_rate
    # self.ident
    return self.initial_learning_rate / tf.math.ceil((step+1)/ (1*(2492 / parameters.batch_size)))

def train_model(datasets, model_to_be_trained, spec_aug_params, audio_aug_params, parse_function, label_passed):

    
    # simply initialize audio loader object for each dataset
    # mandatory parameters:  (1) root of dataset (2) function for extracting filenames 
    # optional parameters: or other custom parameters, like the Bangladesh excel path
    # NOTE: name attribute: to distinguish between datasets when the same audio loader object is used for different datasets, such as antwerp and icbhi that both use IcbhiAudioLoader

    audio_loaders = []
    
    if datasets["Jordan"]: audio_loaders.append(JordanAudioLoader(parameters.jordan_root, default_get_filenames))
    if datasets["Bd"]: audio_loaders.append(BdAudioLoader(parameters.bd_root, bd_get_filenames, parameters.excel_path))
    if datasets["Perch"]: audio_loaders.append(PerchAudioLoader(parameters.perch_root, perch_get_filenames))
    if datasets["Icbhi"]: audio_loaders.append(IcbhiAudioLoader(parameters.icbhi_root, default_get_filenames))
    if datasets["Ant"]: 
        # TODO: pass names?
        ant_loader = IcbhiAudioLoader(parameters.ant_root, default_get_filenames)
        ant_loader.name = "Antwerp"
        audio_loaders.append(ant_loader)
        
    # this functions loads the audios files from the given input, often .wav and .txt files
    # input: [filename1, filename2, ...]
    # output: {'Icbhi': [[audio1, label2, filename1], [audio2, label2, filename2], 'Jordan:' : ... }
    audios_dict = load_audios(audio_loaders)
    # print(audios_dict)
    
    # ths function takes the full audios and prepares its N chunks accordingly
    # by default, it returns samples grouped by patient according to the respective logics of datasets
    # input: [[audio1, label1, filename1], [audio2, label2, filename2], ...]
    # output: [ [all chunks = [audio, label, filename] of all files for patient1], [same for patient 2], ...]
    audios_c_dict = prepare_audios(audios_dict)
    # print(audios_c_dict)

    # NOTE: # Data is grouped by dataset and patient thus far
    # this functions (1) splits each dataset into train and validation, then (2) after split, we don't care about grouping by patient = flatten to list of audios by patients to give a list of audios 
    #  input: Full Dictionary:  {Icbhi: [] -> data grouped by PATIENT, Jordan: [] -> data grouped by PATIENT, ...}
    # output: Training /// Val  dictionary:   {Icbhi: [] -> data organized INDIVIDUALLY, Jordan: [] -> data organized  INDIVIDUALLY} 
    train_audios_c_dict, val_audios_c_dict = split_and_extend(audios_c_dict, parameters.train_test_ratio)
    # NOTE: # Data is only grouped by dataset now
    

    # simplest step: now that everything is ready, we convert to spectrograms! it's the most straightforward step...
    # convert: [audio, label, filename] -> [SPEC, label, filename]
    val_samples = generate_audio_samples(val_audios_c_dict)
    # ... but it's different for training because of augmentation. the following function sets up and merges 2 branches:
    #   1) augment AUDIO and convert to spectrogram
    #   2) convert to spectrogram and augment SPECTROGRAM
    # train_samples, original_training_length = set_up_training_samples(train_audios_c_dict, spec_aug_params, audio_aug_params) 
    train_samples = generate_audio_samples(train_audios_c_dict)
    # train_samples = generate_spec_samples(train_audios_c_dict) # the same as above if no augmentation 
     # NOTE: # Data is NOT LONGER grouped by dataset 

    # samples = train_samples + val_samples
    # np.random.shuffle(samples)
    # train_samples = samples[:int(0.8*len(samples))]
    # val_samples = samples[int(0.8*len(samples)):]

    # for i in range(len(train_samples)):
    #     if train_samples[i][1] == [0,0]:
    #         train_samples[i][1] = 0
    #     elif train_samples[i][1] == [1,0]:
    #         train_samples[i][1] = 1
    #     elif train_samples[i][1] == [0,1]:
    #         train_samples[i][1] = 2
    #     else:
    #         train_samples[i][1] = 3
    
    # for i in range(len(val_samples)):
    #     if val_samples[i][1] == [0,0]:
    #         val_samples[i][1] = 0
    #     elif val_samples[i][1] == [1,0]:
    #         val_samples[i][1] = 1
    #     elif val_samples[i][1] == [0,1]:
    #         val_samples[i][1] = 2
    #     else:
    #         val_samples[i][1] = 3
    
    # parameters.mode = "main"
    # from now on it's cake!
    train_dataset, __, train_labels, __ = create_tf_dataset(train_samples, batch_size=parameters.batch_size, shuffle=True, parse_func=parse_function)
    val_dataset, val_specs, val_labels, val_filenames = create_tf_dataset(val_samples, batch_size=1, shuffle=False, parse_func=parse_function) # keep shuffle = False!
    train_non_pneumonia_nb, train_pneumonia_nb = train_labels.count(0), train_labels.count(1)
    print(train_non_pneumonia_nb)
    print(train_pneumonia_nb)
    print(val_labels.count(0))
    print(val_labels.count(1))

    val_samples_copy = val_samples.copy()
    np.random.shuffle(val_samples_copy)
    samples = val_samples_copy[:25]
    print("-----------------------")
    print_dataset(train_labels, val_labels)

    # weights
    weights = None
    
    # handles metrics, file saving (all the files inside gradcam/, tp/, others/, etc), report writing (report.txt), visualizations, etc
    metrics_callback = NewCallback(val_dataset, val_filenames, target_key="icbhi_score")
    viz_callback = visualizationCallback(samples)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=parameters.job_dir)

    gpus = tf.config.experimental.list_logical_devices('GPU')

    # weights
    weights = None

    if bool(parameters.class_weights):
        print("Initializing weights...")
        weights = class_weight.compute_class_weight(
            "balanced", [0, 1, 2, 3], [metrics_callback.convert(l) for l in train_labels])
        weights = {i: weights[i] for i in range(0, len(weights))}
        print("weights = {}".format(weights))

    for s in [8, 16, 32, 64, 128]:
        # teacher
        teacher = model_to_be_trained(num_outputs=parameters.n_classes, _frontend=leaf_frontend.Leaf(sample_rate=16000, n_filters=80), normalize=True, encoder=None, weights=list(weights.values()), first=0, second=s)
        # teacher = model_to_be_trained(num_outputs=parameters.n_classes, _frontend=leaf_frontend.MelFilterbanks(sample_rate=16000, n_filters=80), normalize=True, encoder=None)
        shape = (None, parameters.audio_length*parameters.sr)
        teacher.build(shape)


        optimizers = [
            tf.keras.optimizers.Adam(
            learning_rate=MyLRSchedule(1e-4, "frontend"), 
            beta_1=0.9, beta_2=0.999, epsilon=parameters.epsilon, decay=parameters.weight_decay, amsgrad=False
            ),
            tf.keras.optimizers.Adam(
            learning_rate=MyLRSchedule(1e-3, "backend"), 
            beta_1=0.9, beta_2=0.999, epsilon=parameters.epsilon, decay=parameters.weight_decay, amsgrad=False
            )
        ]

        optimizers_and_layers = [(optimizers[0], teacher.layers[0]), (optimizers[1], teacher.layers[1:])]
        opt = tfa.optimizers.MultiOptimizer(optimizers_and_layers)

        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=parameters.label_smoothing)

        teacher.compile(
            optimizer=opt,
            loss=loss,
            metrics=[
                'accuracy', 
            ],
        )

        teacher.summary(line_length=110)

        history = teacher.fit(
            train_dataset,
            epochs=parameters.n_epochs,
            verbose=2,
            class_weight=weights,
            callbacks=[metrics_callback, viz_callback, tensorboard_callback]
        )

        plot_metrics(history)
    
    # for i, sample in enumerate(samples):
    #     # print(sample)
    #     audio = sample[0]
    #     audio = np.expand_dims(audio, axis=0)
    #     spec = teacher(audio, False, True)
    #     spec = np.swapaxes(np.squeeze(spec.numpy()), 0, 1)
    #     # if epoch == 0:
    #     #     self.initial_specs.append((spec, sample[1], sample[2]))
    #     # self.final_specs.append((spec, sample[1], sample[2]))
    #         # pair.append((spec, sample[1], sample[2]))
    #     visualize_spec_bis(spec, sr=parameters.sr, dest="spec_{}".format(i))

    #     mel = teacher._mel_frontend(audio)
    #     mel = np.swapaxes(np.squeeze(mel.numpy()), 0, 1)
    #     visualize_spec_bis(mel, sr=parameters.sr, dest="mel_{}".format(i))

    #     mel_bis = frontend.MelFilterbanks(sample_rate=16000, n_filters=80, window_len= 50,window_stride= 5)(audio)
    #     mel_bis = np.swapaxes(np.squeeze(mel_bis.numpy()), 0, 1)
    #     visualize_spec_bis(mel_bis, sr=parameters.sr, dest="mel_bis_{}".format(i))

    #     mel_trd = frontend.Leaf(sample_rate=8000, n_filters=80, window_stride=20)(audio)
    #     mel_trd = np.swapaxes(np.squeeze(mel_trd.numpy()), 0, 1)
    #     visualize_spec_bis(mel_trd, sr=parameters.sr, dest="mel_trd_{}".format(i))

    #     mel_trd = frontend.MelFilterbanks(sample_rate=4000, n_filters=80, max_freq=2000, window_stride=10)(audio)
    #     mel_trd = np.swapaxes(np.squeeze(mel_trd.numpy()), 0, 1)
    #     visualize_spec_bis(mel_trd, sr=parameters.sr, dest="mel_4th_{}".format(i))


    #     visualize_spec_bis(mel, sr=parameters.sr, dest="mel_{}".format(i))

    #     sinc = teacher._sinc_frontend(audio)
    #     sinc = np.swapaxes(np.squeeze(sinc.numpy()), 0, 1)

    #     visualize_spec_bis(sinc, sr=parameters.sr, dest="sinc_{}".format(i))





    
def launch_job(datasets, model, spec_aug_params, audio_aug_params, parse_function, label_passed):
    '''
    parameters: 
    # dictionary:  datasets to use, 
    # model:  imported from modules/models.py, 
    # augmentation parameters:  (No augmentation if empty list is passed), 
    # parse function:  (inside modules/parsers) used to create tf.dataset object in create_tf_dataset
    '''
    # in a given file named train$ (parent/cache folder named train$), we can have multiple jobs (child folders named 1,2,3)
    initialize_job() #  initialize each (child) job inside the file (i.e, creates all the subfolders like tp/tn/gradcam/etc, file saving conventions, etc)
    print("Job dir: {}".format(parameters.job_dir))
    train_model(datasets, model, spec_aug_params, audio_aug_params, parse_function, label_passed)
    

if __name__ == "__main__":
    
    print("Tensorflow Version: {}".format(tf.__version__))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    seed_everything() # seeding np, tf, etc
    arguments = parameters.parse_arguments()
    parameters.init()
    parameters.mode = "pneumonia" if arguments["mode"] == "main" else arguments["mode"]
    parameters.n_classes = 2 if parameters.mode == "cw" else 1
    print(parameters.cache_root)
    print(parameters.mode)
    print(os.path.basename(__file__).split('.')[0])
    parameters.file_dir = os.path.join(parameters.cache_root, parameters.mode, os.path.basename(__file__).split('.')[0])
    parameters.description = arguments["description"]
    print("Description: {}".format(parameters.description))

    parameters.n_epochs = 40

    testing_mode(int(arguments["testing"])) # if true, adds _testing to the cache folder, sets a small number of epochs, smaller dataset, turn off a few settings in the callback, etc
    # works to set up the parent folder (i.e., overwrites if already exists) + works well with initialize_job above
    initialize_file_folder()
    print("-----------------------")


    ###### set up used for spec input models (default)
    parameters.hop_length = 254
    parameters.shape = (128, 311)
    parameters.n_sequences = 9
    
    parameters.early_stopping = False
    parameters.lr_patience = 6
    parameters.batch_size = 32
    parameters.es_patience = 12
    parameters.audio_length = 5
    parameters.weight_decay = 1e-4
    parameters.adaptive_lr = False
    parameters.class_weights = True
    # parameters.mode = "cw"
    parameters.n_classes = 2
    spec_aug_params = []
    audio_aug_params = []
    launch_job({"Bd": 0, "Jordan": 0, "Icbhi": 1, "Perch": 0, "Ant": 0, "SimAnt": 0,}, leaf_model9_model_106, spec_aug_params, audio_aug_params, None, [1,0])
    print("Job dir: {}".format(parameters.job_dir))
    
    # to run another job, add a line to modify whatever parameters, and rerun a launch_job function as many times as you want!
    # parameters.hop_length = 512
    # launch_job({"Bd": 0, "Jordan": 1, "Icbhi": 1, "Perch": 0, "Ant": 0, "SimAnt": 0,}, model9, spec_aug_params, audio_aug_params, spec_parser)
