from .AudioPreparer import AudioPrepaper
from ..main import parameters
import numpy as np
from collections import defaultdict

class BdAudioPreparer(AudioPrepaper):
    pass
    
    def find_label(self, label):
        if label == "NO PEP":
            return 0
        elif label == "PEP":
            return 1
        else:
            return -1
    
    def return_all_samples_by_patient(self): # TODO: standarize across objects, likely by going through every object instead of [0], and using dict

        samples_by_patients = defaultdict(list)
        for chunked_sample in self.chunked_samples:
            key = chunked_sample[0][2].split('/')[-2]
            for i in range(len(chunked_sample)):
                new_filename = chunked_sample[i][2].split('/')[-1].split('.')[0]
                chunked_sample[i][2] = new_filename
            samples_by_patients[key].extend(chunked_sample)
        return list(samples_by_patients.values())