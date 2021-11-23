from .AudioLoader import AudioLoader
from ..main import parameters
import pandas as pd
import os

class JordanAudioLoader(AudioLoader):

    def __init__(self, root, get_filenames):
        super().__init__(root, get_filenames)
        self.root = os.path.join(root, 'Audio_Files')
        self.label_root = root
        self.name = "Jordan"

    def get_patient_id(self, filename):
        return (int(filename.split('_')[0][2:]) - 1)
    
    def read_label(self, _dict, patient_id):
        return _dict[patient_id]
    
    # converting Data_annotation.xlsx to a dictionary to have access to labels for all patients
    def build_label_dict(self, column="E", key="Diagnosis"):
        if self.mode == "cw":
            column = "D"
            key = "Sound type"
        excel_path = os.path.join(self.label_root, "Data_annotation.xlsx")
        label_dict = pd.read_excel(excel_path, engine='openpyxl', usecols=column) #usecols="A:E"
        label_dict = label_dict.to_dict() 
        label_dict = label_dict[key]
        return label_dict
    
    def get_testing_filenames(self):
        return ['BP2_Asthma,E W,P L L R,52,F', 'BP6_Plueral Effusion,I C B,P L R,81,M', 'BP19_heart failure,C,P R U,70,F', 
            'BP26_Lung Fibrosis,Crep,P,90,F', 'BP66_heart failure,Crep,P R L ,43,M', 'BP108_COPD,E W,P R L ,63,M', 'BP100_N,N,P R M,70,F', 'BP18_pneumonia,C,P R U,57,M', 'BP87_N,N,P R M,72,M', 'BP37_pneumonia,Crep,A R L,70,F']