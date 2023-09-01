import types
from ..main import parameters
import os
import librosa
import numpy as np


""" Audio Loader
This is step 1: where we load our samples using 'load_all_samples'. Data usually comes from .wav, and labels from .txt/excel sheets.
We do not apply any slicing yet buts till want to "onboard" the data
Most processes are data-specific, so each data object has its own implementation of 
- get_filenames()
- read_label()
- build_label_dict()
- other functions

EXAMPLE:

self.filenames = ['BP2_Asthma,E W,P L L R,52,F', 'BP6_Plueral Effusion,I C B,P L R,81,M', 'BP19_heart failure,C,P R U,70,F', 'BP26_Lung Fibrosis,Crep,P,90,F', 'BP66_heart failure,Crep,P R L ,43,M', 'BP108_COPD,E W,P R L ,63,M', 'BP100_N,N,P R M,70,F', 'BP18_pneumonia,C,P R U,57,M', 'BP87_N,N,P R M,72,M', 'BP37_pneumonia,Crep,A R L,70,F']

self.samples = {'Jordan': [[array([-2.083e-06, -9.467e-07,  2.297e-06, ...,  2.025e-02,  1.845e-02,
        9.768e-03], dtype=float32), 'Asthma', 'BP2_Asthma,E W,P L L R,52,F'], [array([-5.110e-07, -5.519e-07,  5.078e-07, ..., -2.021e-02, -1.784e-02,
       -9.300e-03], dtype=float32), 'Plueral Effusion', 'BP6_Plueral Effusion,I C B,P L R,81,M'], [array([ 9.815e-08,  9.122e-08, -1.099e-07, ...,  9.202e-03,  7.932e-03,
        4.084e-03], dtype=float32), 'heart failure', 'BP19_heart failure,C,P R U,70,F'], [array([-5.275e-07,  2.542e-07,  6.650e-07, ..., -3.311e-01, -2.833e-01,
       -1.451e-01], dtype=float32), 'Lung Fibrosis', 'BP26_Lung Fibrosis,Crep,P,90,F'], [array([ 1.514e-06,  3.883e-06, -1.697e-06, ..., -2.258e-02, -2.018e-02,
       -1.059e-02], dtype=float32), 'heart failure', 'BP66_heart failure,Crep,P R L ,43,M'], [array([-5.286e-07, -9.805e-07,  5.038e-07, ...,  3.195e-01,  2.741e-01,
        1.406e-01], dtype=float32), 'COPD', 'BP108_COPD,E W,P R L ,63,M'], [array([ 1.834e-06,  1.748e-06, -1.910e-06, ..., -1.992e-02, -1.821e-02,
       -9.656e-03], dtype=float32), 'N', 'BP100_N,N,P R M,70,F'], [array([-8.385e-07, -1.531e-06,  8.435e-07, ..., -7.499e-02, -6.563e-02,
       -3.405e-02], dtype=float32), 'pneumonia', 'BP18_pneumonia,C,P R U,57,M'], [array([ 1.865e-07, -2.514e-07, -2.634e-07, ...,  1.628e-01,  1.399e-01,
        7.185e-02], dtype=float32), 'N', 'BP87_N,N,P R M,72,M'], [array([ 8.769e-07,  8.632e-07, -9.404e-07, ...,  8.454e-02,  7.326e-02,
        3.781e-02], dtype=float32), 'pneumonia', 'BP37_pneumonia,Crep,A R L,70,F']]}
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
        # if testing mode, calls hard-coded filenames for each dataset
        if parameters.testing:
            self.filenames = self.get_testing_filenames()
        # a dictionary of labels for ALL the filenames
        #  For pneumonia, it will be patient_diagnosis.csv for ICBHI, Bangladesh_PCV_onlyStudyPatients for BD and Data_annotation.xlsx for Jordan
        label_dict = self.build_label_dict()
        for filename in self.filenames:
            patient_id = self.get_patient_id(filename)
            audio = self.load_singular_audio(filename)
            # accessing each of the dictonary for individual labels using patient id
            label = self.read_label(label_dict, patient_id)
            # the filename must be stripped of its path (relative or absolute) and extension (.wav)
            # NOTE: except for BD?
            self.samples.append([audio, label, filename])

    def get_patient_id(self, filename):
        pass

    def load_singular_audio(self, filename):
        path = os.path.join(self.root, filename + ".wav")
        data, ___ = librosa.load(path, sr=parameters.sr)
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
