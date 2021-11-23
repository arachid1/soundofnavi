import random
import nlpaug.flow as naf
import nlpaug.augmenter.audio as naa
import nlpaug.augmenter.spectrogram as nas
from audiomentations import *

def augment_wav(data, sr, augmentation, add, ratio, gaussian_noise, vtlp, shift, min_snr, max_snr):
    
    if augmentation:
        print("Augmenting raw audios...")
        k = int(ratio * len(data)) # how many samples we're adding
        for i in range(0, k):
            index = random.randint(0, k-1)
            sample_to_augment = data[index][0]
            if gaussian_noise:
                aug = Compose([AddGaussianSNR(min_SNR=min_snr, max_SNR=max_snr, p=1)])
                sample_to_augment = aug(samples=sample_to_augment, sample_rate=sr)
            if shift:
                aug = naa.ShiftAug(sampling_rate=sr)
                sample_to_augment = aug.augment(sample_to_augment)
            if vtlp:
                aug = naa.VtlpAug(sampling_rate=sr)
                sample_to_augment = aug.augment(sample_to_augment)
            if add:
                data.append((sample_to_augment, data[index][1], data[index][2]))
            else:
                data[index][0] = sample_to_augment        
    return data

def augment_spectrogram(data, sr, augmentation, add, ratio, time_masking, frequency_masking):
    if augmentation:
        print("Augmenting spectrograms...")
        k = int(ratio * len(data)) # how many samples we're adding
        for i in range(0, k):
            index = random.randint(0, k-1)
            sample_to_augment = data[index][0]
            if time_masking:
                aug = naf.Sequential([
                    nas.TimeMaskingAug(),
                ])
                sample_to_augment = aug.augment(sample_to_augment)
            if frequency_masking:
                aug = naf.Sequential([
                    nas.FrequencyMaskingAug(), 
                ])
                sample_to_augment = aug.augment(sample_to_augment)
            if add:
                data.append((sample_to_augment, data[index][1], data[index][2]))
            else:
                data[index][0] = sample_to_augment        
    return data 
