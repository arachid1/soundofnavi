import torch
import pytorch_lightning as pl
import torchaudio
from .main import parameters
import torch.optim as optim
from nnAudio import Spectrogram

from .core_pytorch import *

def gaussian_fn(M, std):
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w

def gkern1d(kernlen=256, std=128):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian_fn(kernlen, std=std) 
    # gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern1d

def gkern2d(kernlen=256, std=128):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian_fn(kernlen, std=std) 
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d

class Model10(pl.LightningModule):
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

class Model9(pl.LightningModule):
    def __init__(self, final_n):
        super(Model9, self).__init__()
        self.spec_layer = Spectrogram.STFT(n_fft=parameters.n_fft, hop_length=parameters.hop_length,  sr=parameters.sr, 
                                            trainable=parameters.train_nn, output_format="Magnitude")
        # self.spec_layer = Spectrogram.MelSpectrogram(n_fft=parameters.n_fft, sr=parameters.sr, n_mels=64, hop_length=parameters.hop_length, 
        #                                     trainable_mel=True, trainable_STFT=True)                        
        self.mel_layer = torchaudio.transforms.MelScaleBis(n_mels=parameters.n_mels, sample_rate=parameters.sr, n_stft=parameters.n_fft//2 + 1, norm="slaney", trainable_mel=parameters.train_mel)           
        self.cwise_filters = gkern(3, 2)
        # print(self.cwise_filters.shape)
        # exit()
        self.cwise_filters = self.cwise_filters.unsqueeze(0)
        self.cwise_filters = self.cwise_filters.unsqueeze(0)
        self.cwise_filters = self.cwise_filters.repeat((parameters.n_mels, 1, 1, 1))
        # print(self.cwise_filters.shape)
        # exit()
        self.cwise_filters = torch.nn.Parameter(self.cwise_filters, requires_grad=True)
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
        x = x.unsqueeze(1)
        x = x.repeat((1, parameters.n_mels, 1, 1))
        print(x.shape)
        print(self.cwise_filters.shape)
        # exit()
        for i in range(x.shape[1]):
            x[:,i,:,:] = torch.nn.functional.conv2d(x[:,i,:,:], weight=self.cwise_filters[i,:,:,:], stride=1, padding="same")
        print(x.shape)
        print(self.cwise_filters.shape)
        exit()
        # x = torch.nn.functional.conv2d(x, weight=self.cwise_filters, stride=1, groups=parameters.n_mels, padding="same")
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

class Model7(pl.LightningModule):
    def __init__(self, final_n):
        super(Model7, self).__init__()
        self.spec_layer = Spectrogram.STFT(n_fft=parameters.n_fft, hop_length=parameters.hop_length,  sr=parameters.sr, 
                                            trainable=parameters.train_nn, output_format="Magnitude")
        # self.spec_layer = Spectrogram.MelSpectrogram(n_fft=parameters.n_fft, sr=parameters.sr, n_mels=64, hop_length=parameters.hop_length, 
        #                                     trainable_mel=True, trainable_STFT=True)                        
        self.mel_layer = torchaudio.transforms.MelScaleBis(n_mels=parameters.n_mels, sample_rate=parameters.sr, n_stft=parameters.n_fft//2 + 1, norm="slaney", trainable_mel=parameters.train_mel)           
        self.cwise_filters = gkern(3, 2)
        self.cwise_filters = self.cwise_filters.unsqueeze(0)
        self.cwise_filters = self.cwise_filters.unsqueeze(0)
        self.cwise_filters = self.cwise_filters.repeat((parameters.n_mels, 1, 1, 1))
        self.cwise_filters = torch.nn.Parameter(self.cwise_filters, requires_grad=True)
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
        x = x.unsqueeze(1)
        x = x.repeat((1, parameters.n_mels, 1, 1))
        x = torch.nn.functional.conv2d(x, weight=self.cwise_filters, stride=1, groups=parameters.n_mels, padding="same")
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



class Model8(Model7):
    def forward(self, x, return_spec=False):
        x = self.spec_layer(x)
        x = self.mel_layer(x)
        x = torchaudio.transforms.AmplitudeToDB()(x)
        if return_spec:
            p = self.mel_layer.return_p()
            return x, p
        x = x.unsqueeze(1)
        x = x.repeat((1, parameters.n_mels, 1, 1))
        # x = torch.nn.functional.conv2d(x, weight=self.cwise_filters, stride=1, groups=parameters.n_mels)
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

class Model6(pl.LightningModule):
    def __init__(self, final_n):
        super(Model6, self).__init__()
        self.spec_layer = Spectrogram.STFT(n_fft=parameters.n_fft, hop_length=parameters.hop_length,  sr=parameters.sr, 
                                            trainable=parameters.train_nn, output_format="Magnitude")
        # self.spec_layer = Spectrogram.MelSpectrogram(n_fft=parameters.n_fft, sr=parameters.sr, n_mels=64, hop_length=parameters.hop_length, 
        #                                     trainable_mel=True, trainable_STFT=True)                        
        self.mel_layer = torchaudio.transforms.MelScaleBis(n_mels=parameters.n_mels, sample_rate=parameters.sr, n_stft=parameters.n_fft//2 + 1, norm="slaney", trainable_mel=parameters.train_mel)           
        self.depth_layer_1 = InvertedResidual_nn(input_channels=1, filters=64, strides=2, expansion_factor=2, kernel_size=(5,5))
        self.depth_layer_2 = InvertedResidual_nn(input_channels=64, filters=64, strides=1, expansion_factor=2, kernel_size=(5,5))
        self.AVGPOOL1 = torch.nn.MaxPool2d(3)
        self.BN1 = torch.nn.BatchNorm2d(64)
        self.DP1 = torch.nn.Dropout(0.5)
        self.depth_layer_3 = InvertedResidual_nn(input_channels=64, filters=128, strides=1, expansion_factor=2, kernel_size=(5,5))
        self.depth_layer_4 = InvertedResidual_nn(input_channels=128, filters=128, strides=1, expansion_factor=2, kernel_size=(5,5))
        self.AVGPOOL2 = torch.nn.MaxPool2d(3)
        self.BN2 = torch.nn.BatchNorm2d(128)
        self.DP2 = torch.nn.Dropout(0.5)
        self.final_neurons = 128*3*14
        self.classifier = torch.nn.Linear(self.final_neurons, final_n)
    
    def training_step(self, batch, batch_idx):
        # print(batch[1])
        pred = self(batch[0])
        # print(pred)
        rounded_pred = torch.round(pred)
        # print(rounded_pred)
        correct = 0
        for i, p in enumerate(rounded_pred):
            if p[0] == batch[1][i][0] and p[1] == batch[1][i][1]:
                correct += 1
        train_acc = correct/(len(batch[1]))
        # print(val_acc)
        loss = torch.nn.functional.binary_cross_entropy(torch.squeeze(pred), batch[1].float())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # print(batch[1])
        pred = self(batch[0])
        # print(pred)
        rounded_pred = torch.round(pred)
        # print(rounded_pred)
        correct = 0
        for i, p in enumerate(rounded_pred):
            if p[0] == batch[1][i][0] and p[1] == batch[1][i][1]:
                correct += 1
        val_acc = correct/(len(batch[1]))
        # print(val_acc)
        loss = torch.nn.functional.binary_cross_entropy(torch.squeeze(pred), batch[1].float())
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

    def forward(self, x, return_spec=False):
        # print("forward")
        # print(x.shape)
        x = self.spec_layer(x)
        x = self.mel_layer(x)
        # x = torch.log(x)
        x = torchaudio.transforms.AmplitudeToDB()(x)
        if return_spec:
            # x = x - x.min()
            # x = x / x.max()
            # x = 2*x - 1
            p = self.mel_layer.return_p()
            return x, p
        x = self.depth_layer_1(x.unsqueeze(1)) # unsqueeze has the purpose of adding a channel dim
        x = self.depth_layer_2(x) # unsqueeze has the purpose of adding a channel dim
        x = self.AVGPOOL1(x)
        x = self.BN1(x)
        x = self.DP1(x)
        x = self.depth_layer_3(x) # unsqueeze has the purpose of adding a channel dim
        x = self.depth_layer_4(x) # unsqueeze has the purpose of adding a channel dim
        x = self.AVGPOOL2(x)
        x = self.BN2(x)
        x = self.DP2(x)
        x = x.view(x.data.size()[0], self.final_neurons)
        
        x = self.classifier(torch.nn.functional.relu(x))
        # print(x.shape)
        return torch.sigmoid(x)

class Model5(pl.LightningModule):
    def __init__(self, final_n):
        super(Model5, self).__init__()
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
        self.classifier = torch.nn.Linear(self.final_neurons, final_n)
    
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
        x = self.DP1(x)
        # x = x.view(x.data.size()[0], -1)
        # print(x.shape)
        x = x.view(x.data.size()[0], self.final_neurons)
        x = self.classifier(torch.nn.functional.relu(x))
        # print(x.shape)
        return torch.sigmoid(x)


#######

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
        print(batch)
        pred = self(batch[0])
        print(pred)
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
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch[0])
        loss = torch.nn.functional.binary_cross_entropy(torch.squeeze(pred), batch[1].float())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
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
