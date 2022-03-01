import torch
import pytorch_lightning as pl
import torchaudio
from .main import parameters
import torch.optim as optim
from nnAudio import Spectrogram

from .core_pytorch import *

class inv_depthwise_model_2(pl.LightningModule):
    def __init__(self, threshold):
        super(inv_depthwise_model_2, self).__init__()
        self.spec_layer = Spectrogram.STFT(n_fft=parameters.n_fft, hop_length=parameters.hop_length,  sr=parameters.sr, 
                                            trainable=parameters.train_nn, output_format="Magnitude")
        # self.spec_layer = Spectrogram.MelSpectrogram(n_fft=parameters.n_fft, sr=parameters.sr, n_mels=64, hop_length=parameters.hop_length, 
        #                                     trainable_mel=True, trainable_STFT=True)                        
        self.mel_layer = torchaudio.transforms.MelScaleBis(n_mels=parameters.n_mels, sample_rate=parameters.sr, n_stft=parameters.n_fft//2 + 1, norm="slaney", trainable_mel=parameters.train_mel)           
        self.depth_layer_1 = InvertedResidual_nn(input_channels=1, filters=256, strides=2, expansion_factor=2, kernel_size=(6,6))
        self.AVGPOOL1 = torch.nn.MaxPool2d(4)
        self.BN1 = torch.nn.BatchNorm2d(256)
        self.DP1 = torch.nn.Dropout(0.6)
        self.final_neurons = 256*15*38
        self.classifier = torch.nn.Linear(self.final_neurons,1)
    
    def training_step(self, batch, batch_idx):
        pred = self(batch[0])
        loss = torch.nn.functional.binary_cross_entropy(torch.squeeze(pred), batch[1].float())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pred = self(batch[0])
        loss = torch.nn.functional.binary_cross_entropy(torch.squeeze(pred), batch[1].float())
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
        
    def configure_optimizers(self):
        """Configure optimizer."""
        return optim.Adam(self.parameters(), lr=parameters.lr)

    def forward(self, x, return_spec=False):
        # print("forward")
        # print(x.shape)
        x = self.spec_layer(x)
        x = self.mel_layer(x)
        # x = torch.log(x)
        x = torchaudio.transforms.AmplitudeToDB()(x)
        if return_spec:
            x = x - x.min()
            x = x / x.max()
            x = 2*x - 1
            p = self.mel_layer.return_p()
            return x, p
        # print(x.shape)
        # print(x.unsqueeze(1).shape)
        x = self.depth_layer_1(x.unsqueeze(1)) # unsqueeze has the purpose of adding a channel dim
        x = self.AVGPOOL1(x)
        x = self.BN1(x)
        # x = x.view(x.data.size()[0], -1)
        # print(x.shape)
        x = x.view(x.data.size()[0], self.final_neurons)
        x = self.DP1(x)
        x = self.classifier(torch.nn.functional.relu(x))
        # print(x.shape)
        return torch.sigmoid(x)

class inv_depthwise_model(pl.LightningModule):
    def __init__(self, threshold):
        super(inv_depthwise_model, self).__init__()
        self.spec_layer = Spectrogram.STFT(n_fft=parameters.n_fft, hop_length=parameters.hop_length,  sr=parameters.sr, 
                                            trainable=parameters.train_nn, output_format="Magnitude")
        # self.spec_layer = Spectrogram.MelSpectrogram(n_fft=parameters.n_fft, sr=parameters.sr, n_mels=64, hop_length=parameters.hop_length, 
        #                                     trainable_mel=True, trainable_STFT=True)                        
        self.mel_layer = torchaudio.transforms.MelScaleBis(n_mels=parameters.n_mels, sample_rate=parameters.sr, n_stft=parameters.n_fft//2 + 1, norm="slaney", trainable_mel=parameters.train_mel)           
        self.depth_layer_1 = InvertedResidual_nn(input_channels=1, filters=32, strides=1, expansion_factor=2, kernel_size=(6,6))
        self.depth_layer_2 = InvertedResidual_nn(input_channels=32, filters=32, strides=1, expansion_factor=2, kernel_size=(6,6))
        self.AVGPOOL1 = torch.nn.MaxPool2d(4)
        self.BN1 = torch.nn.BatchNorm2d(32)
        self.DP1 = torch.nn.Dropout(0.1)
        self.depth_layer_3 = InvertedResidual_nn(input_channels=32, filters=64, strides=1, expansion_factor=2, kernel_size=(6,6))
        self.depth_layer_4 = InvertedResidual_nn(input_channels=64, filters=64, strides=1, expansion_factor=2, kernel_size=(6,6))
        self.AVGPOOL2 = torch.nn.MaxPool2d(4)
        self.BN2 = torch.nn.BatchNorm2d(64)
        self.DP2 = torch.nn.Dropout(0.1)
        self.final_neurons = 64*4*16
        self.classifier = torch.nn.Linear(self.final_neurons,1)
    
    def training_step(self, batch, batch_idx):
        pred = self(batch[0])
        loss = torch.nn.functional.binary_cross_entropy(torch.squeeze(pred), batch[1].float())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pred = self(batch[0])
        loss = torch.nn.functional.binary_cross_entropy(torch.squeeze(pred), batch[1].float())
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
        
    def configure_optimizers(self):
        """Configure optimizer."""
        return optim.Adam(self.parameters(), lr=parameters.lr)

    def forward(self, x, return_spec=False):
        # print("forward")
        # print(x.shape)
        x = self.spec_layer(x)
        x = self.mel_layer(x)
        # x = torch.log(x)
        x = torchaudio.transforms.AmplitudeToDB()(x)
        if return_spec:
            x = x - x.min()
            x = x / x.max()
            x = 2*x - 1
            p = self.mel_layer.return_p()
            return x, p
        # print(x.shape)
        # print(x.unsqueeze(1).shape)
        # print(x.shape)
        x = self.depth_layer_1(x.unsqueeze(1)) # unsqueeze has the purpose of adding a channel dim
        x = self.depth_layer_2(x)
        x = self.AVGPOOL1(x)
        x = self.BN1(x)
        x = self.DP1(x)

        x = self.depth_layer_3(x) # unsqueeze has the purpose of adding a channel dim
        x = self.depth_layer_4(x)
        x = self.AVGPOOL2(x)
        x = self.BN2(x)
        x = self.DP2(x)
        # x = x.view(x.data.size()[0], -1)
        x = x.view(x.data.size()[0], self.final_neurons)
        x = self.classifier(torch.nn.functional.relu(x))
        # print(x.shape)
        return torch.sigmoid(x)

class Model2(pl.LightningModule):

    def __init__(self, threshold):
        super(Model2, self).__init__()
        self.spec_layer = Spectrogram.STFT(n_fft=parameters.n_fft, hop_length=parameters.hop_length,  sr=parameters.sr, 
                                            trainable=parameters.train_nn, output_format="Magnitude")
        # self.spec_layer = Spectrogram.MelSpectrogram(n_fft=parameters.n_fft, sr=parameters.sr, n_mels=64, hop_length=parameters.hop_length, 
        #                                     trainable_mel=True, trainable_STFT=True)                        
        self.mel_layer = torchaudio.transforms.MelScaleBis(n_mels=parameters.n_mels, sample_rate=parameters.sr, n_stft=parameters.n_fft//2 + 1, norm="slaney", trainable_mel=parameters.train_mel)           
        self.CNN_layer1 = torch.nn.Conv2d(1,8, kernel_size=(6, 6))
        self.CNN_layer2 = torch.nn.Conv2d(8, 8, kernel_size=(6, 6))
        self.AVGPOOL1 = torch.nn.MaxPool2d(4)
        self.BN1 = torch.nn.BatchNorm2d(8)
        self.CNN_layer3 = torch.nn.Conv2d(8, 16, kernel_size=(6, 6))
        self.CNN_layer4 = torch.nn.Conv2d(16, 16, kernel_size=(6, 6))
        self.AVGPOOL2 = torch.nn.MaxPool2d(4)
        self.BN2 = torch.nn.BatchNorm2d(16)
        self.final_neurons = 16*4*36
        self.classifier = torch.nn.Linear(self.final_neurons,1)
    
    def training_step(self, batch, batch_idx):
        pred = self(batch[0])
        loss = torch.nn.functional.binary_cross_entropy(torch.squeeze(pred), batch[1].float())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pred = self(batch[0])
        loss = torch.nn.functional.binary_cross_entropy(torch.squeeze(pred), batch[1].float())
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
        
    def configure_optimizers(self):
        """Configure optimizer."""
        return optim.Adam(self.parameters(), lr=parameters.lr)

    def forward(self, x, return_spec=False):
        # print("forward")
        # print(x.shape)
        x = self.spec_layer(x)
        x = self.mel_layer(x)
        # x = torch.log(x)
        x = torchaudio.transforms.AmplitudeToDB()(x)
        if return_spec:
            x = x - x.min()
            x = x / x.max()
            x = 2*x - 1
            p = self.mel_layer.return_p()
            return x, p
        # print(x.shape)
        # print(x.unsqueeze(1).shape)
        # print(x.shape)
        x = self.CNN_layer1(x.unsqueeze(1)) # unsqueeze has the purpose of adding a channel dim
        x = torch.nn.functional.relu(x)
        x = self.CNN_layer2(x)
        x = torch.nn.functional.relu(x)
        x = self.AVGPOOL1(x)
        x = self.BN1(x)
        x = self.CNN_layer3(x)
        x = torch.nn.functional.relu(x)
        x = self.CNN_layer4(x)
        x = torch.nn.functional.relu(x)
        x = self.AVGPOOL2(x)
        x = self.BN2(x)
        # x = x.view(x.data.size()[0], -1)
        # print(x.shape)
        x = x.view(x.data.size()[0], self.final_neurons)
        x = self.classifier(torch.relu(x))
        # print(x.shape)
        return torch.sigmoid(x)

class Model1(pl.LightningModule):
    # exit()

    def __init__(self, threshold):
        super(Model1,self).__init__()
        self.spec_layer = Spectrogram.STFT(n_fft=parameters.n_fft, hop_length=parameters.hop_length, sr=parameters.sr, 
                                            trainable=parameters.train_nn, output_format="Magnitude")
        # self.mel_layer = Spectrogram.MelSpectrogramBis(n_fft=parameters.n_fft, sr=parameters.sr, n_mels=256, hop_length=parameters.hop_length, 
        #                             trainable_mel=True, trainable_STFT=False)   
        self.mel_layer = torchaudio.transforms.MelScaleBis(n_mels=parameters.n_mels, sample_rate=parameters.sr, n_stft=parameters.n_fft//2 + 1, norm="slaney", trainable_mel=parameters.train_mel)
        self.resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)   
        self.threshold = threshold
        self.final_neurons = 1000
        self.classifier = torch.nn.Linear(self.final_neurons,1)
    
    def training_step(self, batch, batch_idx):
        pred = self(batch[0])
        loss = torch.nn.functional.binary_cross_entropy(torch.squeeze(pred), batch[1].float())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pred = self(batch[0])
        loss = torch.nn.functional.binary_cross_entropy(torch.squeeze(pred), batch[1].float())
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
        
    def configure_optimizers(self):
        """Configure optimizer."""
        return optim.Adam(self.parameters(), lr=parameters.lr)
        
    # def on_after_backward(self):
    #     self.spec_layer.wsin.grad[self.threshold:] = 0
    #     self.spec_layer.wcos.grad[self.threshold:] = 0

    def forward(self, x, return_spec=False):
        
        # print("forward")
        # print(x.shape)
        x = self.spec_layer(x)
        x = self.mel_layer(x)
        # x = torch.log(x)
        x = torchaudio.transforms.AmplitudeToDB()(x)
        if return_spec:
            x = x - x.min()
            x = x / x.max()
            x = 2*x - 1
            p = self.mel_layer.return_p()
            return x, p
        x = x[:, None, :, :].repeat(1, 3, 1, 1)
        x = self.resnet_model(x)
        x = self.classifier(torch.relu(x))
        # print(x)
        # # print(x.shape)
        # # print(x.unsqueeze(1).shape)
        # # print(x.shape)
        # x = self.CNN_layer1(x.unsqueeze(1)) # unsqueeze has the purpose of adding a channel dim
        # x = self.CNN_layer2(x)
        # x = self.AVGPOOL1(x)
        # x = self.BN1(x)
        # x = self.CNN_layer3(x)
        # x = self.CNN_layer4(x)
        # x = self.AVGPOOL2(x)
        # x = self.BN2(x)
        # # print(x.shape)
        # # x = x.view(x.data.size()[0], -1)
        # x = x.view(x.data.size()[0], self.final_neurons)
        # x = self.classifier(torch.relu(x))
        # # print(x.shape)
        return torch.sigmoid(x)
