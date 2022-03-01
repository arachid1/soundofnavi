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

from sklearn.metrics import confusion_matrix

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


from collections import Counter
from sklearn.utils import class_weight
import tensorflow as tf
from nnAudio import Spectrogram
from scipy.io import wavfile

import torch
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

class Model(torch.nn.Module):

    def __init__(self):
        super(Model,self).__init__()
        self.spec_layer = Spectrogram.STFT(n_fft=parameters.n_fft, hop_length=parameters.hop_length,  sr=parameters.sr, 
                                            trainable=parameters.train_nn, output_format="Magnitude")
        # self.spec_layer = Spectrogram.MelSpectrogram(n_fft=parameters.n_fft, sr=parameters.sr, n_mels=64, hop_length=parameters.hop_length, 
        #                                     trainable_mel=True, trainable_STFT=True)                                   
        self.CNN_layer1 = torch.nn.Conv2d(1,1, kernel_size=(1, 1))
        self.CNN_layer2 = torch.nn.Conv2d(1,8, kernel_size=(3, 3))
        self.AVGPOOL1 = torch.nn.AvgPool2d(2)
        self.BN1 = torch.nn.BatchNorm2d(8)
        self.CNN_layer3 = torch.nn.Conv2d(8,8, kernel_size=(1,1))
        self.CNN_layer4 = torch.nn.Conv2d(8,16, kernel_size=(3,3))
        self.AVGPOOL2 = torch.nn.AvgPool2d(2)
        self.BN2 = torch.nn.BatchNorm2d(16)
        self.final_neurons = 16*30*76
        self.regressor = torch.nn.Linear(self.final_neurons,1)

    def forward(self, x, return_spec=False):
        # print("forward")
        # print(x.shape)
        x = self.spec_layer(x)
        x = torch.log(x)
        if return_spec:
            return x
        # print(x.shape)
        # print(x.unsqueeze(1).shape)
        # print(x.shape)
        x = self.CNN_layer1(x.unsqueeze(1)) # unsqueeze has the purpose of adding a channel dim
        x = self.CNN_layer2(x)
        x = self.AVGPOOL1(x)
        x = self.BN1(x)
        x = self.CNN_layer3(x)
        x = self.CNN_layer4(x)
        x = self.AVGPOOL2(x)
        x = self.BN2(x)
        # print(x.shape)
        # x = x.view(x.data.size()[0], -1)
        x = x.view(x.data.size()[0], self.final_neurons)
        x = self.regressor(torch.relu(x))
        # print(x.shape)
        return torch.sigmoid(x)

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
    # val
    val_samples = generate_audio_samples(val_audios_c_dict)
    # val_samples = generate_spec_samples(val_audios_c_dict)

    # ... but it's different for training because of augmentation. the following function sets up and merges 2 branches:
    #   1) augment AUDIO and convert to spectrogram
    #   2) convert to spectrogram and augment SPECTROGRAM
    # val
    train_samples = generate_audio_samples(train_audios_c_dict)
    # train_samples, original_training_length = set_up_training_samples(train_audios_c_dict, spec_aug_params, audio_aug_params) 
    # train_samples = generate_spec_samples(train_audios_c_dict) # the same as above if no augmentation 

    ################################ PYTORCH

    print(torch.rand(1, device="cuda:0"))
    torch.cuda.empty_cache()

    _, train_specs, train_labels, train_filenames = create_tf_dataset(train_samples, batch_size=parameters.batch_size, shuffle=True, parse_func=None)
    train_dataset = list(tf.data.Dataset.from_tensor_slices((train_specs, train_labels)).as_numpy_iterator())
    # print(train_dataset[:2])
    # exit()

    positive_cases = [train_dataset[i] for i in range(len(train_dataset)) if train_dataset[i][1] == 1]
    positive_cases = positive_cases 
    negative_cases = [train_dataset[i] for i in range(len(train_dataset)) if train_dataset[i][1] == 0]
    negative_cases = negative_cases[:len(positive_cases)]
    print(len(negative_cases))
    print(len(positive_cases))
    train_dataset = positive_cases + negative_cases

    trainloader = torch.utils.data.DataLoader(dataset=Custom_Dataset(train_dataset),
                                           batch_size=parameters.batch_size,
                                           shuffle=True)

    _, val_specs, val_labels, val_filenames = create_tf_dataset(val_samples, batch_size=1, shuffle=False, parse_func=None)
    val_dataset = list(tf.data.Dataset.from_tensor_slices((val_specs, val_labels)).as_numpy_iterator())
    testloader = torch.utils.data.DataLoader(dataset=Custom_Dataset(val_dataset),
                                           batch_size=parameters.batch_size,
                                           shuffle=False)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # torch.backends.cudnn.benchmark = True

    print(device)
    print(torch.rand(1, device="cuda:0"))

    net = Model()
    net = net.to(device)
    summary(net, (1, 40000))
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


    loss_function = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=parameters.lr, momentum=0.9) 

    original_basis_real = net.spec_layer.wcos.cpu().detach().numpy().squeeze(1)
    original_basis_imag = net.spec_layer.wsin.cpu().detach().numpy().squeeze(1)
    # visualize_spec_bis(outputs, parameters.sr, "test", title="epoch_{}".format(epoch))
    fig = plt.figure(figsize=(20, 10))
    plt.imshow(net.spec_layer.wsin.cpu().detach().numpy().squeeze(1), aspect='auto', origin='lower')
    plt.savefig("weights_original")
    plt.close()

    # index = 0
    # spec_to_track = iter(testloader)[0][index].to(device)
    # # label_to_track = local_labels[index]
    # outputs = net(spec_to_track, return_spec=True)
    # outputs = np.squeeze(outputs.to("cpu").numpy())
    # # print("spec")
    # # print(outputs)
    # visualize_spec_bis(outputs, parameters.sr, "temp/test", title="epoch_{}".format(epoch))

    index = 7
    with torch.no_grad():
            for local_batch, local_labels in testloader:
                print(local_labels)
                spec_to_track = local_batch[index].to(device)
                label_to_track = local_labels[index]
                outputs = net(spec_to_track, return_spec=True)
                old_outputs = np.squeeze(outputs.to("cpu").numpy())
                visualize_spec_bis(old_outputs, parameters.sr, "temp/outputs", title="orig_index_{}_label_{}".format(index, label_to_track))
                break

    old_wcos = net.spec_layer.wcos.cpu()
    old_wsin = net.spec_layer.wsin.cpu()
    
    loss_train = []
    loss_test = []

    print("epoch\ttrain loss\ttest loss")

    for epoch in range(parameters.n_epochs):  # loop over the dataset multiple times
        
        print("Epoch: {}".format(epoch))
        train_correct = 0
        running_loss = 0.0
        loss_train_e = 0.0
        loss_test_e = 0.0

        # if epoch == 10:
        #     for i, param in enumerate(net.parameters()):
        #         param.requires_grad = True

        ### TRAINING ###
        for i, (local_batch, local_labels) in enumerate(trainloader):
            local_labels = local_labels.type(torch.FloatTensor)
            local_labels = local_labels.unsqueeze(1)
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(local_batch)
            loss = loss_function(outputs, local_labels)
            loss.backward()
            optimizer.step()
            loss_train_e += loss.item()
            train_correct += (torch.round(outputs) == local_labels).float().sum()

        loss_train.append(loss_train_e/len(trainloader))

        # intermediary calculations
        # print("Average loss per batch (size {}): {} ".format(parameters.batch_size, loss_train[-1]))
        new_wcos = net.spec_layer.wcos.cpu()
        diff = abs(new_wcos - old_wcos)
        visualize_spec_bis(np.squeeze(diff.detach().numpy()), parameters.sr, "temp/wcos_diff_epoch_{}".format(epoch))
        old_wcos = new_wcos

        new_wsin = net.spec_layer.wsin.cpu()
        diff = abs(new_wsin - old_wsin)
        visualize_spec_bis(np.squeeze(diff.detach().numpy()), parameters.sr, "temp/wsin_diff_epoch_{}".format(epoch))
        old_wsin = new_wsin

        train_accuracy = 100 * train_correct / len(trainloader.dataset)
        print("Train accuracy = {}".format(train_accuracy))
        
        ### VALIDATION ###
        val_correct = 0
        y_true = []
        y_pred = []
        viz_counter = 0
        # with torch.set_grad_enabled(False):
        with torch.no_grad():
            for local_batch, local_labels in testloader:
                if viz_counter == 0 and epoch % 10 == 0:
                    trained_basis_real = net.spec_layer.wcos.cpu().detach().numpy().squeeze(1)
                    trained_basis_imag = net.spec_layer.wsin.cpu().detach().numpy().squeeze(1)
                    # visualize_spec_bis(outputs, parameters.sr, "test", title="epoch_{}".format(epoch))
                    # fig = plt.figure(figsize=(20, 10))
                    # plt.imshow(net.spec_layer.wsin.cpu().detach().numpy().squeeze(1), aspect='auto', origin='lower')
                    # plt.savefig("kernels/weights_wsin_epoch_{}".format(epoch))
                    # plt.imshow(net.spec_layer.wcos.cpu().detach().numpy().squeeze(1), aspect='auto', origin='lower')
                    # plt.savefig("kernels/weights_wcos_epoch_{}".format(epoch))
                    # plt.close()
                    
                    fig, ax = plt.subplots(5,2, figsize=(12,18))
                    cols = ['Original Fourier Kernels', 'Trained Fourier Kernels']
                    rows = np.arange(1,6)
                    for ax_idx, col in zip(ax[0], cols):
                        ax_idx.set_title(col, size=16)
                    for ax_idx, row in zip(ax[:,0], rows):
                        ax_idx.set_ylabel(f'k={row}', size=16)    
                    for i in range(5):
                        ax[i,0].plot(original_basis_real[i+1], 'b')
                        ax[i,1].plot(trained_basis_real[i+1], 'b')
                        ax[i,0].tick_params(labelsize=12)
                        ax[i,1].tick_params(labelsize=12)
                    for i in range(5):
                        ax[i,0].plot(original_basis_imag[i*2+1], 'g')
                        ax[i,1].plot(trained_basis_imag[i*2+1], 'g')
                        ax[i,0].tick_params(labelsize=12)
                        ax[i,1].tick_params(labelsize=12)
                        ax[i,1].legend(['real','imaginary'])
                    plt.savefig("kernels/kernels_epoch_{}".format(epoch))
                    plt.close()

                    # indexes = range(0, 10)
                    # for index in indexes:
                    spec_to_track = local_batch[index].to(device)
                    label_to_track = local_labels[index]
                    outputs = net(spec_to_track, return_spec=True)
                    outputs = np.squeeze(outputs.to("cpu").numpy())
                    visualize_spec_bis(outputs, parameters.sr, "temp/outputs", title="_{}_label_{}_epoch_{}".format(index, label_to_track, epoch))
                    to_viz = outputs - old_outputs 
                    # old_outputs = outputs
                    visualize_spec_bis(to_viz, parameters.sr, "temp/diff", title="_{}_label_{}_epoch_{}".format(index, label_to_track, epoch))

                # Transfer to GPU
                local_labels = local_labels.type(torch.FloatTensor)
                local_labels = local_labels.unsqueeze(1)
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                
                # optimizer.zero_grad()
                outputs = net(local_batch)
                loss = loss_function(outputs, local_labels)
                # loss.requires_grad = False
                # print("val")
                # print(loss)
                # loss.backward()
                # optimizer.step()
                loss_test_e += loss.item()
                outputs = torch.round(outputs)
  
                val_correct += (outputs == local_labels).float().sum()
                y_pred.extend(outputs.detach().cpu().numpy())
                y_true.extend(local_labels.detach().cpu().numpy())
                viz_counter += 1

        loss_test.append(loss_test_e/len(testloader))
        print(' '*100, end='\r')
        print(f"{epoch}\t{loss_train[-1]:.6f}\t{loss_test[-1]:.6f}")

        # testloader_iter = iter(testloader)
        # gradcam_batch, gradcam_labels = next(testloader_iter)
        # l = [module for module in net.modules() if not isinstance(module, nn.Sequential)]
        # target_layers = [l[-1]]

        # cam = GradCAM(model=net, target_layers=target_layers, use_cuda=True)
        
        # n_elements = 5
        # input_tensor = gradcam_batch[:5].to(device)
        # input_tensor = torch.reshape(input_tensor, (5, -1))
        # target_category = gradcam_labels[:5].to(device)
        # print(input_tensor)
        # print(input_tensor.size())
        # print(target_category)
        # print(len(target_category))

        # grayscale_cam = cam(input_tensor=input_tensor)

        # # In this example grayscale_cam has only one image in the batch:
        # rgb_img = None
        # print(grayscale_cam.shape)
        # grayscale_cam = grayscale_cam[0, :]
        # print(grayscale_cam.shape)
        # # visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        # exit()

        val_accuracy =  100*val_correct / len(testloader.dataset)
        cm = confusion_matrix(y_true, y_pred)
        # print("Confusion matrix: {}".format(cm))
        normalized_cm = confusion_matrix(y_true, y_pred, normalize='true')
        # print("Normalized confusion matrix: {}".format(normalized_cm))
        # print("Val accuracy for {} elements = {}".format(len(testloader.dataset), val_accuracy))

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
    parameters.n_fft = 256
    parameters.hop_length = 128
    parameters.train_nn = True

    # general parametes
    parameters.lr = 1e-4
    parameters.n_epochs = 150
    parameters.batch_size = 32
    parameters.audio_length = 5
    parameters.weight_decay = 0

    # augm params
    spec_aug_params = []
    audio_aug_params = []
    
    launch_job({"Bd": 0, "Jordan": 1, "Icbhi": 1, "Perch": 0, "Ant": 0, "SimAnt": 0,}, mixednet, spec_aug_params, audio_aug_params, spec_parser)

    # to run another job, add a line to modify whatever parameters, and rerun a launch_job function as many times as you want!