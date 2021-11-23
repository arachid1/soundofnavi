from .AudioLoader import AudioLoader
from ..main import parameters
import os
import pandas as pd
import types


ICBHI_FILES = ['220_1b2_Al_mc_LittC2SE', '220_1b1_Tc_mc_LittC2SE', '217_1b1_Tc_sc_Meditron', 
            '152_1b1_Al_sc_Meditron', '179_1b1_Al_sc_Meditron', '179_1b1_Tc_sc_Meditron', '191_2b1_Pl_mc_LittC2SE', '191_2b1_Pr_mc_LittC2SE', 
            '191_2b2_Tc_mc_LittC2SE', '226_1b1_Al_sc_Meditron', '226_1b1_Ll_sc_Meditron', '226_1b1_Pl_sc_LittC2SE', '107_2b5_Pr_mc_AKGC417L', 
            '130_1p3_Tc_mc_AKGC417L', '130_2b3_Tc_mc_AKGC417L', '138_1p2_Lr_mc_AKGC417L', '147_2b4_Pl_mc_AKGC417L']

class IcbhiAudioLoader(AudioLoader):

    def __init__(self, root, get_filenames):
        super().__init__(root, get_filenames)
        self.name = "Icbhi"
        if self.mode == "cw":
            self.build_label_dict = types.MethodType(build_cw_label_dict, self) 
            self.read_label = types.MethodType(read_cw_label, self) 
        else:
            self.build_label_dict = types.MethodType(build_pneumonia_label_dict, self) 
            self.read_label = types.MethodType(read_pneumonia_label, self) 

    def get_patient_id(self, filename):
        # if self.mode == "cw":
        #     return filename
        return int(filename[:3])
    
    def read_label(self, _dict, patient_id):
        pass

    def build_label_dict(self):
        pass
    
    def get_testing_filenames(self):
        return ['220_1b2_Al_mc_LittC2SE', '220_1b1_Tc_mc_LittC2SE', '217_1b1_Tc_sc_Meditron', 
            '152_1b1_Al_sc_Meditron', '179_1b1_Al_sc_Meditron', '179_1b1_Tc_sc_Meditron', '191_2b1_Pl_mc_LittC2SE', '191_2b1_Pr_mc_LittC2SE', 
            '191_2b2_Tc_mc_LittC2SE', '226_1b1_Al_sc_Meditron', '226_1b1_Ll_sc_Meditron', '226_1b1_Pl_sc_LittC2SE', '107_2b5_Pr_mc_AKGC417L', 
            '130_1p3_Tc_mc_AKGC417L', '130_2b3_Tc_mc_AKGC417L', '138_1p2_Lr_mc_AKGC417L', '147_2b4_Pl_mc_AKGC417L']
    
def read_cw_label(self, _dict, patient_id):
    return _dict[patient_id]

def read_pneumonia_label(self, _dict, patient_id):
    return _dict.loc[patient_id]

def build_pneumonia_label_dict(self):
    label_dict = pd.read_csv(os.path.join(
        self.root, "patient_diagnosis.csv"), names=['Diagnosis'], delimiter=',', index_col=0)
    return label_dict

def build_cw_label_dict(self):
    annotated_dict = {}
    for s in self.filenames:
        a = Extract_Annotation_Data(s, self.root)
        annotated_dict[s] = a
    return annotated_dict

def Extract_Annotation_Data(file_name, root):
    recording_annotations = pd.read_csv(os.path.join(
        root, file_name + '.txt'), names=['Start', 'End', 'Crackles', 'Wheezes'], delim_whitespace=True)
    return recording_annotations