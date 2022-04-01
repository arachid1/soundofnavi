# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from .core import *
# from tensorflow.keras import layers
# from tensorflow.keras.regularizers import l2
# from kapre.composed import get_melspectrogram_layer
# from kapre.time_frequency import STFT, Magnitude, Phase, MagnitudeToDecibel, ApplyFilterbank
# from ..main import parameters

import torch
import pytorch_lightning as pl
import torchaudio
from ..main import parameters
import torch.optim as optim
from nnAudio import Spectrogram

from .core_pytorch import *

    
class ModelA(pl.LightningModule):
    def __init__(self, final_n, device):
        super(Model10, self).__init__()
        self.spec_layer = Spectrogram.STFT(n_fft=parameters.n_fft, hop_length=parameters.hop_length,  sr=parameters.sr, 
                                            trainable=parameters.train_nn, output_format="Magnitude")
        # self.spec_layer = Spectrogram.MelSpectrogram(n_fft=parameters.n_fft, sr=parameters.sr, n_mels=64, hop_length=parameters.hop_length, 
        #                                     trainable_mel=True, trainable_STFT=True)                        
        self.mel_layer = torchaudio.transforms.MelScaleBis(n_mels=parameters.n_mels, sample_rate=parameters.sr, n_stft=parameters.n_fft//2 + 1, norm="slaney", trainable_mel=parameters.train_mel)           
        # self.device = device
        # self.cwise_filters = gkern2d(3, 2)
        # self.cwise_filters = self.cwise_filters.unsqueeze(0)
        # self.cwise_filters = self.cwise_filters.unsqueeze(0)
        self.cwise_filters = []
        for i in range(4, 20):
            for y in range(1, 9):
                # f = torch.nn.Parameter(gkern2d(i, y), requires_grad=True)
                f = gkern2d(i, y)
                f = f.unsqueeze(0)
                f = f.unsqueeze(0)
                f = torch.nn.Parameter(f, requires_grad=True)
                self.cwise_filters.append(f)
        # print(len(self.cwise_filters))
        # self.cwise_filters = torch.Tensor(self.cwise_filters)
        # self.cwise_filters = self.cwise_filters.repeat((parameters.n_mels, 1, 1, 1))
        # self.cwise_filters = torch.nn.Parameter(self.cwise_filters, requires_grad=True)
        self.depth_layer_1 = InvertedResidual_nn(input_channels=parameters.n_mels, filters=parameters.n_mels, strides=2, expansion_factor=2, kernel_size=(5,5))
        # self.depth_layer_2 = InvertedResidual_nn(input_channels=128, filters=128, strides=1, expansion_factor=2, kernel_size=(5,5))
        self.AVGPOOL1 = torch.nn.AvgPool2d(3)
        self.BN1 = torch.nn.BatchNorm2d(parameters.n_mels)
        self.DP1 = torch.nn.Dropout(0.5)
        self.depth_layer_3 = InvertedResidual_nn(input_channels=parameters.n_mels, filters=parameters.n_mels, strides=1, expansion_factor=2, kernel_size=(5,5))
        self.AVGPOOL2 = torch.nn.AvgPool2d(3)
        self.BN2 = torch.nn.BatchNorm2d(parameters.n_mels)
        self.DP2 = torch.nn.Dropout(0.5)
        self.final_neurons = parameters.n_mels 
        self.classifier = torch.nn.Linear(self.final_neurons, final_n)
    
    def return_filters(self):
        return self.cwise_filters

    def training_step(self, batch, batch_idx):
        pred = self(batch[0])
        rounded_pred = torch.round(pred)
        correct = 0
        for i, p in enumerate(rounded_pred):
            if p[0] == batch[1][i][0] and p[1] == batch[1][i][1]:
                correct += 1
        train_acc = correct/(len(batch[1]))
        loss = torch.nn.functional.binary_cross_entropy(pred, batch[1].float())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pred = self(batch[0])
        rounded_pred = torch.round(pred)
        correct = 0
        for i, p in enumerate(rounded_pred):
            if p[0] == batch[1][i][0] and p[1] == batch[1][i][1]:
                correct += 1
        val_acc = correct/(len(batch[1]))
        loss = torch.nn.functional.binary_cross_entropy(pred, batch[1].float())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
 
    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = optim.Adam(self.parameters(), lr=parameters.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=3, verbose=True)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'val_loss'
       }

    def forward(self, x, return_spec=False, return_filt_spec=False):
        x = self.spec_layer(x)
        x = self.mel_layer(x)
        x = torchaudio.transforms.AmplitudeToDB()(x)
        x = x - x.min()
        x = x / x.max()
        x = 2*x - 1
        if return_spec:
            p = self.mel_layer.return_p()
            return x, p
        # print(x.shape)
        # x = x.unsqueeze(1)
        # print(x.shape)
        # x = x.repeat((1, parameters.n_mels, 1, 1))
        # print(x.shape)
        out = []
        for i in range(len(self.cwise_filters)):
            temp = x
            # print(temp.shape)
            temp = temp.unsqueeze(1)
            # print(temp.shape)
            # print(self.cwise_filters[i].shape)
            out.append(torch.nn.functional.conv2d(temp, weight=self.cwise_filters[i].cuda(), padding="same"))
            # out.append(torch.nn.functional.conv2d(x[:,i,:,:], weight=self.cwise_filters[i], padding="same"))
            # x[:,i,:,:] = x_convolved
            # exit()
        x = torch.cat(out, dim=1)
        # print(x.shape)
        # exit()
        if return_filt_spec:
            return x
        x = self.depth_layer_1(x) # unsqueeze has the purpose of adding a channel dim
        # x = self.depth_layer_2(x) # unsqueeze has the purpose of adding a channel dim
        x = self.AVGPOOL1(x)
        x = self.BN1(x)
        x = self.DP1(x)
        x = self.depth_layer_3(x) # unsqueeze has the purpose of adding a channel dim
        # x = self.depth_layer_4(x) # unsqueeze has the purpose of adding a channel dim
        x = self.AVGPOOL2(x)
        x = self.BN2(x)
        x = self.DP2(x)
        # x = x.view(x.data.size()[0], self.final_neurons)
        x = torch.mean(x, (2, 3))
        x = self.classifier(torch.nn.functional.relu(x))
        x = torch.sigmoid(x)
        return x