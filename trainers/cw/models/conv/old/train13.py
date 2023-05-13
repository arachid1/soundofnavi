import sys
# sys.path.insert(0, 'main/models/conv')
from modules.main import parameters
from modules.main.helpers import *
from modules.main.global_helpers import visualize_spec_bis
from modules.audio_loader.JordanAudioLoader import JordanAudioLoader
from modules.audio_loader.IcbhiAudioLoader import IcbhiAudioLoader
from modules.audio_loader.BdAudioLoader import BdAudioLoader
from modules.audio_loader.PerchAudioLoader import PerchAudioLoader
from modules.audio_loader.helpers import default_get_filenames, bd_get_filenames, perch_get_filenames
from modules.spec_generator.SpecGenerator import SpecGenerator
from modules.callbacks.NewCallback import NewCallback
# from modules.models import mixednet, time_series_model, kapre_model
from modules.parse_functions.parse_functions import spec_parser, generate_timed_spec
from modules.augmenter.functions import stretch_time, shift_pitch 

from pytorch_lightning.loggers import TensorBoardLogger
# from modules.pytorch_core import *
from modules.models.modelA_pytorch import ModelA

from sklearn.metrics import confusion_matrix

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_lightning.callbacks import Callback

from collections import Counter
from sklearn.utils import class_weight
import tensorflow as tf
from nnAudio import Spectrogram
from scipy.io import wavfile

import torch
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

import os, shutil
# from .models import *

class Custom_Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, _dataset):
        self.dataset = _dataset

    def __getitem__(self, index):
        example, target = self.dataset[index]
        return np.array(example), target

    def __len__(self):
        return len(self.dataset)

class MetricTracker(Callback):
    def __init__(self, folder):
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.folder = folder
    
    # def on_validation_batch_end(self, trainer, module):
    #     self.val_losses.append(trainer.logged_metrics["val_loss"])
    
    def on_train_epoch_end(self, trainer, module):
        self.train_losses.append(trainer.logged_metrics["train_loss"])
        self.val_losses.append(trainer.logged_metrics["val_loss"])
        self.val_accuracies.append(trainer.logged_metrics["val_acc"])

    def on_train_end(self, trainer, pl_module):
        plt.plot(torch.range(1, len(self.train_losses)), self.train_losses)
        plt.savefig("{}/train_loss.png".format(self.folder)) 
        plt.close()
        plt.plot(torch.range(1, len(self.val_losses)), self.val_losses)
        plt.savefig("{}/val_loss.png".format(self.folder)) 
        plt.close()
        plt.plot(torch.range(1, len(self.val_accuracies)), self.val_accuracies)
        plt.savefig("{}/accs.png".format(self.folder)) 
        plt.close()

def train_model(datasets, model_to_be_trained, spec_aug_params, audio_aug_params, parse_function):

    
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
    # val_samples = generate_spec_samples(val_audios_c_dict)

    # ... but it's different for training because of augmentation. the following function sets up and merges 2 branches:
    #   1) augment AUDIO and convert to spectrogram
    #   2) convert to spectrogram and augment SPECTROGRAM
    train_samples = generate_audio_samples(train_audios_c_dict)
    # train_samples, original_training_length = set_up_training_samples(train_audios_c_dict, spec_aug_params, audio_aug_params) 
    # train_samples = generate_spec_samples(train_audios_c_dict) # the same as above if no augmentation 

    ################################ PYTORCH

    print(torch.rand(1, device="cuda:0"))
    torch.cuda.empty_cache()

    _, train_specs, train_labels, train_filenames = create_tf_dataset(train_samples, batch_size=parameters.batch_size, shuffle=True, parse_func=None)
    train_dataset = list(tf.data.Dataset.from_tensor_slices((train_specs, train_labels)).as_numpy_iterator())

    # positive_cases = [train_dataset[i] for i in range(len(train_dataset)) if train_dataset[i][1] == 1]
    # # positive_cases = positive_cases 
    # negative_cases = [train_dataset[i] for i in range(len(train_dataset)) if train_dataset[i][1] == 0]
    # # negative_cases = negative_cases[:len(positive_cases)]
    # print(len(negative_cases))
    # print(len(positive_cases))
    # train_dataset = positive_cases + negative_cases

    trainloader = torch.utils.data.DataLoader(dataset=Custom_Dataset(train_dataset),
                                           batch_size=parameters.batch_size,
                                           num_workers=8,
                                           shuffle=True)

    _, val_specs, val_labels, val_filenames = create_tf_dataset(val_samples, batch_size=1, shuffle=False, parse_func=None)
    val_dataset = list(tf.data.Dataset.from_tensor_slices((val_specs, val_labels)).as_numpy_iterator())
    valloader = torch.utils.data.DataLoader(dataset=Custom_Dataset(val_dataset),
                                           batch_size=parameters.batch_size,
                                           num_workers=16,
                                           shuffle=False)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # torch.backends.cudnn.benchmark = True

    print(device)
    print(torch.rand(1, device="cuda:0"))

    # models
    # epochs
    # train_mel

    d = parameters.job_dir

    # if os.path.isdir(d):
    #     shutil.rmtree(d)
    #     print('contents inside {} removed'.format(d))
    # os.mkdir(d)
    
    ind = 0
    for m in [ModelA]:
        for e in [3]:
            for t in [True]:
                folder = "{}/temp{}/".format(d, ind)
                os.mkdir(folder)
                ind += 1
                parameters.train_mel = t
                parameters.n_epochs = e
                model = m(2, device).to(device)

                no_train_outs = []
                ps = []
                indexes = [0, 1]
                with torch.no_grad():
                        for local_batch, local_labels in valloader:
                            print(local_labels)
                            for index in indexes:
                                spec_to_track = local_batch[index].to(device)
                                label_to_track = local_labels[index]
                                
                                outputs, p = model(spec_to_track, return_spec=True)
                                no_train_out = np.squeeze(outputs.to("cpu").numpy())
                                visualize_spec_bis(p.cpu().numpy(), parameters.sr, "{}/p_orig_{}_".format(folder, index))
                                visualize_spec_bis(no_train_out, parameters.sr, "{}/outputs_orig_{}_".format(folder, index), title="index_{}_label_{}".format(index, label_to_track))
                                no_train_outs.append(no_train_out)
                                ps.append(p)
                                
                                if index == indexes[-1]:
                                    nt_filt_outputs = model(spec_to_track, return_filt_spec=True)
                                    nt_filt_outputs = np.squeeze(nt_filt_outputs.to("cpu").numpy())
                            break

                # original_basis_real = model.spec_layer.wcos.cpu().detach().numpy().squeeze(1)
                # original_basis_imag = model.spec_layer.wsin.cpu().detach().numpy().squeeze(1)
                # original_weight = model.spec_layer.wsin.detach().cpu()

                # logger = TensorBoardLogger(save_dir=parameters.job_dir, version=1, name="lightning_logs")
                logger = None
                trainer = pl.Trainer(max_epochs=parameters.n_epochs, logger=logger, callbacks=[MetricTracker(folder)], gpus=1, progress_bar_refresh_rate=1000)

                # lr_finder = trainer.tuner.lr_find(model)
                # # Results can be found in
                # lr_finder.results
                # # Plot with
                # fig = lr_finder.plot(suggest=True)
                # plt.savefig(fig)
                # # fig.show()
                # # Pick point based on plot, or get suggestion
                # new_lr = lr_finder.suggestion()
                # print(new_lr)
                # # update hparams of the model
                # model.hparams.lr = new_lr
                # exit()

                trainer.fit(model, trainloader, val_dataloaders=valloader)

                model = model.to(device)

                with torch.no_grad():
                        for local_batch, local_labels in valloader:
                            local_labels = local_labels.type(torch.FloatTensor)
                            local_labels = local_labels.unsqueeze(1)
                            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                            i = 0
                            for index in indexes:
                                spec_to_track = local_batch[index]
                                label_to_track = local_labels[index]
                                outputs, p = model(spec_to_track, return_spec=True)
                                last_epoch_out = np.squeeze(outputs.to("cpu").numpy())
                                to_viz = last_epoch_out - no_train_outs[i]
                                p_to_viz = p.cpu().numpy() - ps[i].cpu().numpy()
                                visualize_spec_bis(to_viz, parameters.sr, "{}/outputs_diff_{}".format(folder, index))
                                visualize_spec_bis(p_to_viz, parameters.sr, "{}/p_diff_{}".format(folder, index))
                                visualize_spec_bis(p.cpu().numpy(), parameters.sr, "{}/p_end_{}_".format(folder, index))
                                visualize_spec_bis(last_epoch_out, parameters.sr, "{}/outputs_end_{}_".format(folder, index), title="index_{}_label_{}".format(index, label_to_track))
                                i += 1

                                if index == indexes[-1]:
                                    t_filt_outputs = model(spec_to_track, return_filt_spec=True)
                                    t_filt_outputs = np.squeeze(t_filt_outputs.to("cpu").numpy())
                                    for z in range(t_filt_outputs.shape[0]):
                                        tfo = t_filt_outputs[i]
                                        ntfo = nt_filt_outputs[i]
                                        diff = tfo - ntfo
                                        # visualize_spec_bis(tfo, parameters.sr, "{}/z_index_{}_tfo_change_{}".format(folder, index, z))
                                        # visualize_spec_bis(ntfo, parameters.sr, "{}/z_index_{}_ntfo_change_{}".format(folder, index, z))
                                        visualize_spec_bis(diff, parameters.sr, "{}/z_index_{}_diff_change_{}".format(folder, index, z))

                            break


                print(parameters.job_dir)


def launch_job(datasets, model, spec_aug_params, audio_aug_params, parse_function):
    '''
    parameters: 
    # dictionary:  datasets to use, 
    # model:  imported from modules/models.py, 
    # augmentation parameters:  (No augmentation if empty list is passed), 
    # parse function:  (inside modules/parsers) used to create tf.dataset object in create_tf_dataset
    '''
    # in a given file named train$ (parent/cache folder named train$), we can have multiple jobs (child folders named 1,2,3)
    initialize_job() #  initialize each (child) job inside the file (i.e, creates all the subfolders like tp/tn/gradcam/etc, file saving conventions, etc)
    train_model(datasets, model, spec_aug_params, audio_aug_params, parse_function)

if __name__ == "__main__":
    
    print("Tensorflow Version: {}".format(tf.__version__))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    seed_everything() # seeding np, tf, etc
    arguments = parameters.parse_arguments()
    parameters.init()
    parameters.mode = "cw"
    parameters.file_dir = os.path.join(parameters.cache_root, parameters.mode, os.path.basename(__file__).split('.')[0])
    parameters.description = arguments["description"]
    testing_mode(int(arguments["testing"])) # if true, adds _testing to the cache folder, sets a small number of epochs, smaller dataset, turn off a few settings in the callback, etc
    # works to set up the parent folder (i.e., overwrites if already exists) + works well with initialize_job above
    initialize_file_folder()
    print("-----------------------")
    
    ###### set up used for spec input models (default)
    # parameters.n_fft = 5
    # parameters.hop_length = 254
    # parameters.shape = (128, 311)
    # parameters.n_sequences = 9
    # parameters.batch_size = 8

    ###### set up used for audio input models
    parameters.mode = "cw"
    parameters.n_fft = 512
    parameters.hop_length = 128
    parameters.n_mels = 128
    parameters.train_nn = False
    parameters.train_mel = True

    # general parametes
    parameters.lr = 1e-3
    parameters.n_epochs = 25 
    parameters.batch_size = 2
    parameters.audio_length = 5
    parameters.step_size  = 2.5
    parameters.weight_decay = 0

    # augm params
    spec_aug_params = []
    audio_aug_params = []
    
    launch_job({"Bd": 0, "Jordan": 0, "Icbhi": 1, "Perch": 1, "Ant": 1, "SimAnt": 1,}, None, spec_aug_params, audio_aug_params, spec_parser)

    # to run another job, add a line to modify whatever parameters, and rerun a launch_job function as many times as you want!