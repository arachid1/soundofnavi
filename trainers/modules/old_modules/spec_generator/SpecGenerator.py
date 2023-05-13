import types
from ..main import parameters
from ..main.global_helpers import visualize_spec
import librosa
import numpy as np
import os

class SpecGenerator:
    def __init__(self, samples, name):
        self.samples = samples
        self.name = name
        self.subfolder = "spec"
        self.visualize_spec = types.MethodType(visualize_spec, self) 
        self.spec_samples = []
    
    def generate_all_samples(self):
        counter = 0
        for sample in self.samples:
            spec_sample = self.generate_singular_sample(sample)
            # print(spec_sample)
            # print(spec_sample[0].shape)
            # print(parameters.job_id)
            # dest = subfolder_job + "/specs" + name + "_spec"
            # dest = os.path.join(parameters.job_dir, str(parameters.job_id), self.subfolder, spec_sample[2])
            # print(dest)
            # self.visualize_spec(spec_sample[0], parameters.sr, dest)
            # counter += 1
            # if counter == 3:
            #     exit()
            self.spec_samples.append(spec_sample)

    def generate_singular_sample(self, sample):
        audio = sample[0]
        label = sample[1]
        filename = sample[2]
        spec = self.generate_mel_spec(audio)
        return [spec, label, filename]
    
    def generate_mel_spec(self, audio):
        return librosa.power_to_db(
            librosa.feature.melspectrogram(audio, sr=parameters.sr, n_fft=parameters.n_fft, 
                hop_length=parameters.hop_length, n_mels=parameters.n_mels, center=False), 
            ref=np.max)
        # print(parameters.n_fft)
        # print(parameters.hop_length)
        # return librosa.feature.melspectrogram(audio, sr=parameters.sr, n_fft=parameters.n_fft, 
        #         hop_length=parameters.hop_length, n_mels=parameters.n_mels, power=2, center=False)
        # print("done")

    def return_all_samples(self):
        return self.spec_samples
    
    def visualize_spec(self):
        pass
