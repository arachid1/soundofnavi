import types
from ..main import parameters
import os 
import librosa
from ..main.global_helpers import visualize_audio
import numpy as np


""" Audio Loader

"""
class AudioLoader:
    def __init__(self, root, get_filenames):
        self.root = root
        self.get_filenames = types.MethodType(get_filenames, self) 
        self.mode = parameters.mode
        self.filenames = []
        self.samples = []
        self.name = None
    
    def load_all_samples(self):
        # filenames without .wav
        self.filenames = self.get_filenames() 
        # print(self.filenames[:20])
        if parameters.testing:
            self.filenames = self.get_testing_filenames()
        # a dictionary of labels for ALL the filenames
        #  For pneumonia, it will be patient_diagnosis.csv for ICBHI, Bangladesh_PCV_onlyStudyPatients for BD and Data_annotation.xlsx for Jordan
        label_dict = self.build_label_dict()
        for filename in self.filenames:
            print("inside")
            print(filename)
            patient_id = self.get_patient_id(filename)
            print(patient_id)
            audio = self.load_singular_audio(filename)
            # accessing each of the dictonary for individual labels using patient id 
            label = self.read_label(label_dict, patient_id)
            print(label)
            self.samples.append([audio, label, filename])
    
    def get_patient_id(self, filename):
        pass

    def load_singular_audio(self, filename):
        path = os.path.join(self.root, filename + '.wav')
        data, ___ = librosa.load(path, parameters.sr)
        return data

    def read_label(self, _dict, id):
        pass

    def load_label(self):
        pass

    def get_filenames(self):
        pass

    def get_testing_filenames(self):
        pass

    def return_all_samples(self):
        return self.samples
    
    def return_name(self):
        return self.name