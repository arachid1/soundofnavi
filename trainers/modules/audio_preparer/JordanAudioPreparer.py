from .AudioPreparer import AudioPrepaper
from ..main import parameters
import numpy as np

class JordanAudioPreparer(AudioPrepaper):
    pass
    
    def find_label(self, label):
        if parameters.mode == "cw":
            new_label = [0, 0]
            labels = label.split()
            if 'C' in labels:
                new_label[0] = 1
            if 'W' in labels:
                new_label[1] = 1
            # print(new_label)
            return new_label
        if label == 'pneumonia':
            return 1
        else:
            return 0
    
    # we have 113 files and look for a specific filter type, so we can parse through all possible files and collect the present ones with the right FT
    def return_all_samples_by_patient(self):

        samples_by_patients = []
        filter_type = "BP"
        for i in range(1, 113):
            data_type = "{}{}_".format(filter_type, i)
            patient_samples = []
            count = 0
            for s in self.chunked_samples:
                if (s[0][2].startswith(data_type)):
                    patient_samples.extend(s)
                count += 1
            # no samples have been 
            if len(patient_samples) == 0:
                # print("Jordan patient N{} has no samples (JordanAudioPreparer)".format(i))
                continue
            samples_by_patients.append(patient_samples) #TODO: instead of doing at [0], squeeze (& standarize across objects)
        self.chunked_samples = samples_by_patients
        return self.chunked_samples
