import sys
sys.path.insert(0, 'main/models/conv')
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
from modules.models import mixednet, model9, time_series_model, kapre_model
from modules.parse_functions.parse_functions import spec_parser, generate_timed_spec
from modules.augmenter.functions import stretch_time, shift_pitch 

from pytorch_lightning.loggers import TensorBoardLogger
# from modules.pytorch_core import *
from modules.models_pytorch import inv_depthwise_model_2_cw, inv_depthwise_model_3_cw

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
        self.folder = folder
    
    def on_validation_batch_end(self, trainer, module):
        self.val_losses.append(trainer.logged_metrics["val_loss"])
    
    def on_train_epoch_end(self, trainer, module):
        self.train_losses.append(trainer.logged_metrics["train_loss_epoch"])

    def on_train_end(self, trainer, pl_module):
        plt.plot(torch.range(1, len(self.train_losses)), self.train_losses)
        plt.savefig("{}/losses.png".format(self.folder)) 

# class Model2(pl.LightningModule):

#     def __init__(self, threshold):
#         super(Model2, self).__init__()
#         self.spec_layer = Spectrogram.STFT(n_fft=parameters.n_fft, hop_length=parameters.hop_length,  sr=parameters.sr, 
#                                             trainable=parameters.train_nn, output_format="Magnitude")
#         # self.spec_layer = Spectrogram.MelSpectrogram(n_fft=parameters.n_fft, sr=parameters.sr, n_mels=64, hop_length=parameters.hop_length, 
#         #                                     trainable_mel=True, trainable_STFT=True)                        
#         self.mel_layer = torchaudio.transforms.MelScaleBis(n_mels=parameters.n_mels, sample_rate=parameters.sr, n_stft=parameters.n_fft//2 + 1, norm="slaney", trainable_mel=parameters.train_mel)           
#         self.CNN_layer1 = torch.nn.Conv2d(1,8, kernel_size=(6, 6))
#         self.CNN_layer2 = torch.nn.Conv2d(8, 8, kernel_size=(6, 6))
#         self.AVGPOOL1 = torch.nn.MaxPool2d(4)
#         self.BN1 = torch.nn.BatchNorm2d(8)
#         self.CNN_layer3 = torch.nn.Conv2d(8, 16, kernel_size=(6, 6))
#         self.CNN_layer4 = torch.nn.Conv2d(16, 16, kernel_size=(6, 6))
#         self.AVGPOOL2 = torch.nn.MaxPool2d(4)
#         self.BN2 = torch.nn.BatchNorm2d(16)
#         self.final_neurons = 16*4*36
#         self.classifier = torch.nn.Linear(self.final_neurons,1)
    
#     def training_step(self, batch, batch_idx):
#         pred = self(batch[0])
#         loss = torch.nn.functional.binary_cross_entropy(torch.squeeze(pred), batch[1].float())
#         self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         pred = self(batch[0])
#         loss = torch.nn.functional.binary_cross_entropy(torch.squeeze(pred), batch[1].float())
#         self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         return loss
        
#     def configure_optimizers(self):
#         """Configure optimizer."""
#         return optim.Adam(self.parameters(), lr=parameters.lr)

#     def forward(self, x, return_spec=False):
#         # print("forward")
#         # print(x.shape)
#         x = self.spec_layer(x)
#         x = self.mel_layer(x)
#         # x = torch.log(x)
#         x = torchaudio.transforms.AmplitudeToDB()(x)
#         if return_spec:
#             x = x - x.min()
#             x = x / x.max()
#             x = 2*x - 1
#             p = self.mel_layer.return_p()
#             return x, p
#         # print(x.shape)
#         # print(x.unsqueeze(1).shape)
#         # print(x.shape)
#         x = self.CNN_layer1(x.unsqueeze(1)) # unsqueeze has the purpose of adding a channel dim
#         x = torch.nn.functional.relu(x)
#         x = self.CNN_layer2(x)
#         x = torch.nn.functional.relu(x)
#         x = self.AVGPOOL1(x)
#         x = self.BN1(x)
#         x = self.CNN_layer3(x)
#         x = torch.nn.functional.relu(x)
#         x = self.CNN_layer4(x)
#         x = torch.nn.functional.relu(x)
#         x = self.AVGPOOL2(x)
#         x = self.BN2(x)
#         # x = x.view(x.data.size()[0], -1)
#         # print(x.shape)
#         x = x.view(x.data.size()[0], self.final_neurons)
#         x = self.classifier(torch.relu(x))
#         # print(x.shape)
#         return torch.sigmoid(x)

# class Model(pl.LightningModule):
#     exit()

#     def __init__(self, threshold):
#         super(Model,self).__init__()
#         self.spec_layer = Spectrogram.STFT(n_fft=parameters.n_fft, hop_length=parameters.hop_length, sr=parameters.sr, 
#                                             trainable=parameters.train_nn, output_format="Magnitude")
#         # self.mel_layer = Spectrogram.MelSpectrogramBis(n_fft=parameters.n_fft, sr=parameters.sr, n_mels=256, hop_length=parameters.hop_length, 
#         #                             trainable_mel=True, trainable_STFT=False)   
#         self.mel_layer = torchaudio.transforms.MelScaleBis(n_mels=parameters.n_mels, sample_rate=parameters.sr, n_stft=parameters.n_fft//2 + 1, norm="slaney", trainable_mel=parameters.train_mel)
#         self.resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)   
#         self.threshold = threshold
#         self.final_neurons = 1000
#         self.classifier = torch.nn.Linear(self.final_neurons,1)
    
#     def training_step(self, batch, batch_idx):
#         pred = self(batch[0])
#         loss = torch.nn.functional.binary_cross_entropy(torch.squeeze(pred), batch[1].float())
#         self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         pred = self(batch[0])
#         loss = torch.nn.functional.binary_cross_entropy(torch.squeeze(pred), batch[1].float())
#         self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         return loss
        
#     def configure_optimizers(self):
#         """Configure optimizer."""
#         return optim.Adam(self.parameters(), lr=parameters.lr)
        
#     # def on_after_backward(self):
#     #     self.spec_layer.wsin.grad[self.threshold:] = 0
#     #     self.spec_layer.wcos.grad[self.threshold:] = 0

#     def forward(self, x, return_spec=False):
        
#         # print("forward")
#         # print(x.shape)
#         x = self.spec_layer(x)
#         x = self.mel_layer(x)
#         # x = torch.log(x)
#         x = torchaudio.transforms.AmplitudeToDB()(x)
#         if return_spec:
#             x = x - x.min()
#             x = x / x.max()
#             x = 2*x - 1
#             p = self.mel_layer.return_p()
#             return x, p
#         x = x[:, None, :, :].repeat(1, 3, 1, 1)
#         x = self.resnet_model(x)
#         x = self.classifier(torch.relu(x))
#         # print(x)
#         # # print(x.shape)
#         # # print(x.unsqueeze(1).shape)
#         # # print(x.shape)
#         # x = self.CNN_layer1(x.unsqueeze(1)) # unsqueeze has the purpose of adding a channel dim
#         # x = self.CNN_layer2(x)
#         # x = self.AVGPOOL1(x)
#         # x = self.BN1(x)
#         # x = self.CNN_layer3(x)
#         # x = self.CNN_layer4(x)
#         # x = self.AVGPOOL2(x)
#         # x = self.BN2(x)
#         # # print(x.shape)
#         # # x = x.view(x.data.size()[0], -1)
#         # x = x.view(x.data.size()[0], self.final_neurons)
#         # x = self.classifier(torch.relu(x))
#         # # print(x.shape)
#         return torch.sigmoid(x)

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
                                           shuffle=False)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # torch.backends.cudnn.benchmark = True

    print(device)
    print(torch.rand(1, device="cuda:0"))

    threshold = "none"

    # models
    # epochs
    # train_mel

    d = "temp/"
    if os.path.isdir(d):
        shutil.rmtree(d)
        print('contents inside {} removed'.format(d))
    os.mkdir(d)
    
    ind = 0
    for m in [inv_depthwise_model_2_cw]:
        for e in [50,100]:
            for t in [True]:
                folder = "temp/{}/".format(ind)
                os.mkdir(folder)
                ind += 1
                parameters.train_mel = t
                parameters.n_epochs = e
                model = m(threshold).to(device)
                # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
                # first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
                # first_conv_layer.extend(list(model.features))  
                # model.features= nn.Sequential(*first_conv_layer ) 
        #       summary(model, (1, 40000))
                # exit()
                # for layer in net.layers():
                #     print(layer)
                # print(len(list(net.parameters())))
                # print(list(net.parameters)[:3])

                # for i, param in enumerate(net.parameters()):
                #     if i == 2:
                #         break
                #     param.requires_grad = False
                # net.spec_layer.params.requires_grad = False
                # net.spec_layer.bias.requires_grad = False

                # loss_function = nn.BCELoss()
                # optimizer = optim.SGD(model.parameters(), lr=parameters.lr, momentum=0.9) 


                # # visualize_spec_bis(outputs, parameters.sr, "test", title="epoch_{}".format(epoch))
                # fig = plt.figure(figsize=(20, 10))
                # plt.imshow(net.spec_layer.wsin.cpu().detach().numpy().squeeze(1), aspect='auto', origin='lower')
                # plt.savefig("weights_original")
                # plt.close()

                # index = 0
                # spec_to_track = iter(valloader)[0][index].to(device)
                # # label_to_track = local_labels[index]
                # outputs = net(spec_to_track, return_spec=True)
                # outputs = np.squeeze(outputs.to("cpu").numpy())
                # # print("spec")
                # # print(outputs)
                # visualize_spec_bis(outputs, parameters.sr, "temp/test", title="epoch_{}".format(epoch))

                index = 7
                with torch.no_grad():
                        for local_batch, local_labels in valloader:
                            print(local_labels)
                            # for i in range(len(local_labels)):
                            #     if local_labels[i] == 1:
                            #         index = i
                            #         break
                            spec_to_track = local_batch[index].to(device)
                            label_to_track = local_labels[index]
                            outputs, p = model(spec_to_track, return_spec=True)
                            no_train_out = np.squeeze(outputs.to("cpu").numpy())
                            visualize_spec_bis(p.cpu().numpy(), parameters.sr, "{}/orig_p_{}_".format(folder, threshold))
                            visualize_spec_bis(no_train_out, parameters.sr, "{}/orig_outputs_{}_".format(folder, threshold), title="index_{}_label_{}".format(index, label_to_track))
                            break

                # original_basis_real = model.spec_layer.wcos.cpu().detach().numpy().squeeze(1)
                # original_basis_imag = model.spec_layer.wsin.cpu().detach().numpy().squeeze(1)
                # original_weight = model.spec_layer.wsin.detach().cpu()

                logger = TensorBoardLogger(save_dir=parameters.job_dir, version=1, name="lightning_logs")
                trainer = pl.Trainer(max_epochs=parameters.n_epochs, logger=logger, callbacks=[MetricTracker(folder)], gpus=1)

                trainer.fit(model, trainloader)

                # changed_weight = model.spec_layer.wsin.detach().cpu()

                # check if bin 0-20 are still the same after training
                # print(torch.equal(original_weight[:20],changed_weight[:20]))
                # It should return False∂

                # check if bin 20-1025 are still the same after training
                # print(torch.equal(original_weight[20:],changed_weight[20:]))

                model = model.to(device)

                with torch.no_grad():
                        for local_batch, local_labels in valloader:
                            local_labels = local_labels.type(torch.FloatTensor)
                            local_labels = local_labels.unsqueeze(1)
                            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                            spec_to_track = local_batch[index]
                            label_to_track = local_labels[index]
                            outputs, p = model(spec_to_track, return_spec=True)
                            last_epoch_out = np.squeeze(outputs.to("cpu").numpy())
                            visualize_spec_bis(p.cpu().numpy(), parameters.sr, "{}/end_p_{}_".format(folder, threshold))
                            visualize_spec_bis(last_epoch_out, parameters.sr, "{}/end_outputs_{}_".format(folder, threshold), title="index_{}_label_{}".format(index, label_to_track))
                            break

                to_viz = last_epoch_out - no_train_out 
                visualize_spec_bis(to_viz, parameters.sr, "{}/diff_{}".format(folder, threshold))

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
    parameters.mode = "pneumonia"
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
    parameters.n_epochs = 5
    parameters.batch_size = 8
    parameters.audio_length = 5
    parameters.step_size  = 2.5
    parameters.weight_decay = 0

    # augm params
    spec_aug_params = []
    audio_aug_params = []
    
    launch_job({"Bd": 0, "Jordan": 0, "Icbhi": 1, "Perch": 1, "Ant": 1, "SimAnt": 1,}, mixednet, spec_aug_params, audio_aug_params, spec_parser)

    # to run another job, add a line to modify whatever parameters, and rerun a launch_job function as many times as you want!