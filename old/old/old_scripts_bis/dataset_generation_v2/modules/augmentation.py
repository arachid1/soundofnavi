import random
import nlpaug.flow as naf
import nlpaug.augmenter.audio as naa
import nlpaug.augmenter.spectrogram as nas
from audiomentations import *
from .helpers import *
import librosa

def augment_wav(data, sr, augmentation, add, ratio, gaussian_noise, vtlp, shift, min_snr, max_snr):
    
    if augmentation:
        print("Augmenting raw audios...")
        print("Original length: {}".format(len(data)))
        k = int(ratio * len(data)) # how many samples we're adding
        new_index = 100
        for i in range(0, k):
            index = random.randint(0, k-1)
            sample_to_augment = data[index][0]
            if gaussian_noise:
                aug = Compose([AddGaussianSNR(min_SNR=min_snr, max_SNR=max_snr, p=1)])
                sample_to_augment = aug(samples=sample_to_augment, sample_rate=sr)
            if shift:
                aug = naa.ShiftAug(sampling_rate=sr)
                sample_to_augment = aug.augment(sample_to_augment)
            # if vtlp:
            #     aug = naa.VtlpAug(sampling_rate=sr)
            #     sample_to_augment = aug.augment(sample_to_augment)
            if add:
                # print(data[index])
                # exit()
                # new_index = data[index][3] + 100
                data.append((sample_to_augment, data[index][1], data[index][2], new_index, data[index][4]))
                new_index = new_index + 1
            else:
                data[index][0] = sample_to_augment
        print("Augmented length: {}".format(len(data)))        
    return data


def augment_audios(cycles, sr, audio_length, wav_params):
    for cycle in cycles:
        # print(cycle)
        if bool(wav_params['PITCH_SHIFTING']) == True:
            aug = naa.PitchAug(sampling_rate=sr)
            cycle[0] = aug.augment(cycle[0])
        if bool(wav_params['TIME_STRETCHING']) == True:
            factor = random.uniform(0.6, 1.2)
            cycle[0] = librosa.effects.time_stretch(cycle[0], factor)
            # print("start")
            # print(cycle[0])
            # print(np.sum(cycle[0]))
            desired_length = int(sr * audio_length)
            if len(cycle[0]) >= desired_length:
                # print("higher length: {}".format(len(cycle[0])))
                cycle[0] = cycle[0][:desired_length]
            else:
                # print("lower length: {}".format(len(cycle[0])))
                cycle[0] = generate_padded_samples(cycle[0], desired_length)
            # print("after")
            # print(cycle[0])
            # print(np.sum(cycle[0]))
            # print(len(cycle[0]))
        if bool(wav_params['DRC']) == True:
            pass
    return cycles

def augment_spectrogram(data, sr, augmentation, add, ratio, time_masking, frequency_masking):

    if augmentation:
        print("Augmenting spectrograms...")
        print("Original length: {}".format(len(data)))
        k = int(ratio * len(data)) # how many samples we're adding
        new_index = 100
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
                # new_index = random.randint(100, 100000)
                data.append((sample_to_augment, data[index][1], data[index][2], new_index, data[index][4]))
                new_index = new_index + 1
                # data.append((sample_to_augment, data[index][1], data[index][2]))
            else:
                data[index][0] = sample_to_augment        
        print("Augmented length: {}".format(len(data)))
    return data 
