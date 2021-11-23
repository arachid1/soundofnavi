# from .modules.helpers import *
# from .modules.generators import *
# from .modules.metrics import *
# from .modules.callbacks import *
# from .modules.pneumonia import *
# from .modules.parse_functions import *
# from .modules.augmentation import *
# from .core import *
import sys
sys.path.insert(0, '/home/alirachidi/classification_algorithm/trainers/main/models/conv')
from modules.main import parameters
from modules.main.helpers import *

from modules.audio_loader.JordanAudioLoader import JordanAudioLoader
from modules.audio_loader.IcbhiAudioLoader import IcbhiAudioLoader
from modules.audio_loader.BdAudioLoader import BdAudioLoader
from modules.audio_loader.PerchAudioLoader import PerchAudioLoader
from modules.audio_loader.helpers import default_get_filenames, bd_get_filenames, perch_get_filenames

# from .modules.audio_preparer.helpers import *

from modules.spec_generator.SpecGenerator import SpecGenerator

# from .modules.augmenter.helpers import *

from modules.callbacks.NewCallback import NewCallback

from modules.models import mixednet, model9, time_series_model, audio_model

from modules.parse_functions.parse_functions import generate_spec, generate_timed_spec

from modules.augmenter.functions import stretch_time, shift_pitch 

# from .modules.main.global_helpers import visualize_spec, pad_sample
from collections import Counter
from sklearn.utils import class_weight
import tensorflow as tf

from nnAudio import Spectrogram
from scipy.io import wavfile
import torch
import torch.optim as optim
import torch.nn as nn


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
        self.spec_layer = Spectrogram.STFT(n_fft=2048, freq_bins=None,
                                           hop_length=512, window='hann',
                                           freq_scale='no', center=True,
                                           pad_mode='reflect', fmin=50,
                                           fmax=6000, sr=22050, trainable=True,
                                           output_format='Magnitude')
        
        print(self.spec_layer)

        self.n_bins = 2048 // 2 + 1 # number of bins
        # self.epsilon = parameters.epsilon
        regions = 3 # seems to be some time variable
        m = 1 # output size for last layer (which should be 1)
        

        # Creating CNN Layers
        self.CNN_freq_kernel_size=(128,1)
        self.CNN_freq_kernel_stride=(2,1)
        k_out = 128
        k2_out = 256
        self.CNN_freq = nn.Conv2d(1,k_out,
                                kernel_size=self.CNN_freq_kernel_size,stride=self.CNN_freq_kernel_stride)
        print(self.CNN_freq)
        self.CNN_time = nn.Conv2d(k_out,k2_out,
                                kernel_size=(1,regions),stride=(1,1))

        self.region_v = 1 + (self.n_bins-self.CNN_freq_kernel_size[0])//self.CNN_freq_kernel_stride[0]
        self.linear = torch.nn.Linear(k2_out*self.region_v, m, bias=False)
        # self.linear = torch.nn.Linear(k_out, m, bias=False)
        # print(self.linear)

    def forward(self,x):
        z = self.spec_layer(x)
        print(z.shape)
        z = torch.log(z+parameters.epsilon)
        print(z.shape)
        print(z.unsqueeze(1).shape)
        z2 = torch.relu(self.CNN_freq(z.unsqueeze(1)))
        print(z2.shape)
        z3 = torch.relu(self.CNN_time(z2))
        print(z3.shape)
        y = self.linear(torch.relu(torch.flatten(z2,1)))
        print(y.shape)
        return torch.sigmoid(y)

def train_model(datasets, model_to_be_trained, spec_aug_params, audio_aug_params, parse_function):

    # full audios
    audio_loaders = []

    # TODO: pass names and testing files as elements
    print(parameters.jordan_root)
    if datasets["Jordan"]: audio_loaders.append(JordanAudioLoader(parameters.jordan_root, default_get_filenames))
    # if datasets["Bd"]: audio_loaders.append(BdAudioLoader(parameters.bd_root, bd_get_filenames, parameters.excel_path))
    if datasets["Perch"]: audio_loaders.append(PerchAudioLoader(parameters.perch_root, perch_get_filenames))
    if datasets["Icbhi"]: audio_loaders.append(IcbhiAudioLoader(parameters.icbhi_root, default_get_filenames))
    if datasets["Ant"]: 
        ant_loader = IcbhiAudioLoader(parameters.ant_root, default_get_filenames)
        ant_loader.name = "Antwerp"
        audio_loaders.append(ant_loader)
        
    # if datasets["SimAnt"]: audio_loaders.append(PerchAudioLoader(bd_root, bd_get_filenames, excel_path, mode=parameters.mode))

    audios_dict = load_audios(audio_loaders)

    # audio chunks
    audios_c_dict = prepare_audios(audios_dict)

    # split and extend
    train_audios_c_dict, val_audios_c_dict = split_and_extend(audios_c_dict, parameters.train_test_ratio)

    # val
    val_samples = generate_audio_samples(val_audios_c_dict)

    # train
    train_samples = generate_audio_samples(train_audios_c_dict)

    ################################ PYTORCH

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    _, train_specs, train_labels, train_filenames = create_tf_dataset(train_samples, batch_size=parameters.batch_size, shuffle=True, parse_func=parse_function)
    _, val_specs, val_labels, val_filenames = create_tf_dataset(val_samples, batch_size=1, shuffle=False, parse_func=parse_function)

    train_dataset = list(tf.data.Dataset.from_tensor_slices((train_specs, train_labels)).as_numpy_iterator())
    train_loader = torch.utils.data.DataLoader(dataset=Custom_Dataset(train_dataset),
                                           batch_size=parameters.batch_size,
                                           shuffle=True)

    val_dataset = list(tf.data.Dataset.from_tensor_slices((val_specs, val_labels)).as_numpy_iterator())
    val_loader = torch.utils.data.DataLoader(dataset=Custom_Dataset(val_dataset),
                                           batch_size=parameters.batch_size,
                                           shuffle=True)
    
    net = TorchModel()
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=parameters.lr, momentum=0.9) 

    for epoch in range(parameters.n_epochs):  # loop over the dataset multiple times
        

        # TRAINING
        running_loss = 0.0
        for local_batch, local_labels in train_loader:
            # get the inputs; data is a list of [inputs, labels]
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(local_batch)
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
    
    exit()


    # train_non_pneumonia_nb, train_pneumonia_nb = train_labels.count(0), train_labels.count(1)
    # print("-----------------------")
    # print_dataset(train_labels, val_labels)

    # # weights
    # weights = None

    # if bool(parameters.class_weights):
    #     print("Initializing weights...")
    #     weights = class_weight.compute_class_weight(
    #         "balanced", [0, 1], [l for l in train_labels if l == 0 or l == 1])
    #     weights = {i: weights[i] for i in range(0, len(weights))}
    #     print("weights = {}".format(weights))
    
    

    # # callbacks
    # metrics_callback = NewCallback(val_dataset, val_filenames)

    # gpus = tf.config.experimental.list_logical_devices('GPU')

    # # model setting
    # model = model_to_be_trained(**parameters.return_model_params())

    # model.summary(line_length=110)

    # if len(gpus) > 1:
    #     print("You are using 2 GPUs while the code is set up for one only.")
    #     exit()

    # # training
    # model.fit(
    #     train_dataset,
    #     epochs=parameters.n_epochs,
    #     verbose=2,
    #     class_weight=weights,
    #     callbacks=[metrics_callback]
    # )

    # model.save(parameters.job_dir + "/model_{}.h5".format(parameters.n_epochs))

def launch_job(datasets, model, spec_aug_params, audio_aug_params, parse_function):
    initialize_job()
    train_model(datasets, model, spec_aug_params, audio_aug_params, parse_function)

if __name__ == "__main__":
    
    seed_everything()
    arguments = parameters.parse_arguments()
    # print(arguments)
    parameters.init()
    # parameters.mode = "cw"
    parameters.file_dir = os.path.join(parameters.cache_root, parameters.mode, os.path.basename(__file__).split('.')[0])
    parameters.description = arguments["description"]
    if int(arguments["testing"]):
        parameters.file_dir += "_testing"
        parameters.n_epochs = 2
        parameters.train_test_ratio = 0.5
        parameters.testing = 1
        parameters.description = "testing"

    initialize_file_folder()
    print("-----------------------")
    spec_aug_params = []
    audio_aug_params = []
    parameters.shape = (80000, )
    for t_fb in [True, False]:
        for t_d in [True, False]:
            parameters.trainable_fb = t_fb
            parameters.to_decibel = t_d
            launch_job({"Icbhi": 1, "Jordan": 1, "Bd": 1, "Ant": 0, "Perch": 0}, audio_model, spec_aug_params, audio_aug_params, None)
    # parameters.hop_length = 254
    # parameters.shape = (128, 315)
    # parameters.n_sequences = 9
    # spec_aug_params = []
    # audio_aug_params = []
    # launch_job(time_series_model, spec_aug_params, audio_aug_params, generate_timed_spec)

