import sys
sys.path.insert(0, 'main/models/conv')
from modules.main import parameters
from modules.main.helpers import *

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

from collections import Counter
from sklearn.utils import class_weight
import tensorflow as tf

from nnAudio import Spectrogram
from scipy.io import wavfile
import torch
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary



# from .models import *

class Custom_Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, _dataset):
        self.dataset = _dataset

    def __getitem__(self, index):
        example, target = self.dataset[index]
        return np.array(example), target

    def __len__(self):
        return len(self.dataset)


class TorchModel(torch.nn.Module):
    def __init__(self):
        super(TorchModel, self).__init__()
        # Getting Mel Spectrogram on the fly
        self.spec_layer = Spectrogram.STFT(n_fft=parameters.n_fft, freq_bins=None,
                                           hop_length=parameters.hop_length, window='hann',
                                           freq_scale='no', center=True,
                                           pad_mode='reflect', 
                                           #fmin=50,fmax=6000, 
                                           sr=parameters.sr, trainable=True,
                                           output_format='Magnitude'
                                           )
        
        print(self.spec_layer)

        pool_size = 2
        self.n_bins = 2048 // 2 + 1 # number of bins
        m = 1 # output size for last layer (which should be 1)
        

        # Block 1
        self.CNN_freq_kernel_size=(16,1)
        self.CNN_freq_kernel_stride=(2,1)
        k_out = 32
        k2_out = 64
        regions = 15 # seems to be some time variable
        
        self.CNN_freq = nn.Conv2d(1,k_out,
                                kernel_size=self.CNN_freq_kernel_size,stride=self.CNN_freq_kernel_stride)
        print("frq layer")
        print(self.CNN_freq)
        self.CNN_time = nn.Conv2d(k_out,k2_out,
                                kernel_size=(1,regions),stride=(1,1))
        print("time layer")
        print(self.CNN_time)
        self.AvgPool = nn.AvgPool2d(pool_size)
        self.bn = nn.BatchNorm2d(k2_out)

        # Block 2
        self.CNN_freq_kernel_size=(16,1)
        self.CNN_freq_kernel_stride=(2,1)
        k_out = 64
        k2_out = 128
        regions = 15 # seems to be some time variable

        self.CNN_freq_2 = nn.Conv2d(k_out,k_out,
                                kernel_size=self.CNN_freq_kernel_size,stride=self.CNN_freq_kernel_stride)
        print("frq layer")
        print(self.CNN_freq_2)
        self.CNN_time_2 = nn.Conv2d(k_out,k2_out,
                                kernel_size=(1,regions),stride=(1,1))
        print("time layer")
        print(self.CNN_time_2)
        self.AvgPool_2 = nn.AvgPool2d(pool_size)
        self.bn_2 = nn.BatchNorm2d(k2_out)

        self.region_v = 1 + (self.n_bins-self.CNN_freq_kernel_size[0])//self.CNN_freq_kernel_stride[0]
        print("expected linear input length after")
        print(k2_out*self.region_v)
        # self.linear = torch.nn.Linear(k2_out*self.region_v, m, bias=False)
        self.linear = torch.nn.Linear(235008, m, bias=False)
        # self.linear = torch.nn.Linear(k_out, m, bias=False)
        print("linear layer")
        print(self.linear)

    def forward(self,x):
        # print("forward")
        # print("ff: input")
        # print(x.shape)
        z = self.spec_layer(x)
        # print(z.shape)
        z = torch.log(z+parameters.epsilon)

        z2 = torch.relu(self.CNN_freq(z.unsqueeze(1)))
        # print("ff: first frq conv")
        # print(z2.shape)
        z3 = torch.relu(self.CNN_time(z2))
        # print("ff: second time conv")
        # print(z3.shape)
        z3 = self.AvgPool(z3)
        # print("ff: pool")
        # print(z3.shape)
        z3 = self.bn(z3)
        # print("ff: bn")
        # print(z3.shape)

        z4 = torch.relu(self.CNN_freq_2(z3))
        # print("ff: third frq conv")
        # print(z4.shape)
        z5 = torch.relu(self.CNN_time_2(z4))
        # print("ff: fourth time conv")
        # print(z5.shape)
        z5 = self.AvgPool_2(z5)
        # print("ff: pool")
        # print(z5.shape)
        z5 = self.bn_2(z5)
        # print("ff: bn")
        # print(z5.shape)


        y = self.linear(torch.relu(torch.flatten(z5,1)))
        return torch.sigmoid(y)

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

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    _, train_specs, train_labels, train_filenames = create_tf_dataset(train_samples, batch_size=parameters.batch_size, shuffle=True, parse_func=None)
    _, val_specs, val_labels, val_filenames = create_tf_dataset(val_samples, batch_size=1, shuffle=False, parse_func=None)

    train_dataset = list(tf.data.Dataset.from_tensor_slices((train_specs, train_labels)).as_numpy_iterator())
    train_loader = torch.utils.data.DataLoader(dataset=Custom_Dataset(train_dataset),
                                           batch_size=parameters.batch_size,
                                           shuffle=True)

    val_dataset = list(tf.data.Dataset.from_tensor_slices((val_specs, val_labels)).as_numpy_iterator())
    val_loader = torch.utils.data.DataLoader(dataset=Custom_Dataset(val_dataset),
                                           batch_size=parameters.batch_size,
                                           shuffle=True)
    
    net = TorchModel()
    summary(net, (1, 80000))
    net = net.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=parameters.lr, momentum=0.9) 

    for epoch in range(parameters.n_epochs):  # loop over the dataset multiple times

        # TRAINING
        running_loss = 0.0
        for i, (local_batch, local_labels) in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            local_labels = local_labels.type(torch.FloatTensor)
            local_labels = local_labels.unsqueeze(1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(local_batch)

            # print(outputs)
            # print(local_labels)
            loss = criterion(outputs, local_labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        
        print('Finished training for epoch {}'.format(epoch))
        
        # VALIDATION
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in val_loader:
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                # Model computations
                outputs = net(local_batch)

        print('Finished validation for epoch {}'.format(epoch))

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
    parameters.hop_length = 254
    parameters.shape = (128, 311)
    parameters.n_sequences = 9
    spec_aug_params = [
        ["mixup", {"quantity" : 0.2, "no_pad" : False, "label_one" : 0, "label_two" : 1, "minval" : 0.3, "maxval" : 0.7}]
    ]
    audio_aug_params = [
        ["augmix", {"quantity" : 0.2, "label": -1, "no_pad" : False, "minval" : 0.3, "maxval" : 0.7, "aug_functions": [shift_pitch, stretch_time]}]
    ]
    launch_job({"Bd": 0, "Jordan": 1, "Icbhi": 1, "Perch": 0, "Ant": 0, "SimAnt": 0,}, mixednet, spec_aug_params, audio_aug_params, spec_parser)

    # to run another job, add a line to modify whatever parameters, and rerun a launch_job function as many times as you want!