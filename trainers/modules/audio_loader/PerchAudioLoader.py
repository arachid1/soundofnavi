from .AudioLoader import AudioLoader
from ..main import parameters
import os
import pandas as pd
import types
from .IcbhiAudioLoader import build_cw_label_dict, read_cw_label

class PerchAudioLoader(AudioLoader):

    def __init__(self, root, get_filenames):
        super().__init__(root, get_filenames)
        self.name = "Perch"
        if self.mode == "cw":
            self.build_label_dict = types.MethodType(build_cw_label_dict, self) 
            self.read_label = types.MethodType(read_cw_label, self) 

    def get_patient_id(self, filename):
        if self.mode == "cw":
            return filename
        return int(filename[:3])
    
    def read_label(self, _dict, patient_id):
        # print(_dict)
        # print(patient_id)
        # exit()
        return _dict.loc[patient_id]

    def build_label_dict(self):
        pass
    
    def get_testing_filenames(self):
        return ['Z02666-08.wav_F', 'Z02671-01.wav_F', 'Z02716-02.wav_F', 'B00687-04.wav_F', 'B00761-07.wav_F', 'B00809-02.wav_F', 'Z02736-03.wav_F', 'Z03085-03.wav_F', 'Z03105-04.wav_F', 'G01145-05.wav_F',
            'N17802-03.wav_F','B00858-08.wav_F', 'Z02828-03.wav_F', 'Z02854-08.wav_F', 'Z02820-08.wav_F', 'Z03002-04.wav_F', 'B00915-03.wav_F', 'B00876-06.wav_F', 'Z02961-01.wav_F', 'Z02961-07.wav_F', 'T14644-06.wav_F']
    

# def read_cw_label(self, _dict, patient_id):
#     # for item, key in _dict.items():
#     #     print(item)
#     #     print(key)
#     #     exit()
#     # print(_dict)
#     # print(patient_id)
#     return _dict[patient_id]

# def read_pneumonia_label(self, _dict, patient_id):
#     return _dict.loc[patient_id]

# def build_pneumonia_label_dict(self):
#     label_dict = pd.read_csv(os.path.join(
#         self.root, "patient_diagnosis.csv"), names=['Diagnosis'], delimiter=',', index_col=0)
#     return label_dict

# def build_cw_label_dict(self):
#     # filenames = [s.split('.')[0] # TODO: Rewrite this as part of get_filenames
#     #                 for s in os.listdir(path=self.root) if '.wav' in s]
#     filenames = self.get_filenames()
#     # ASSERT
#     print("Number of Files: {}".format(len(filenames)))

#     # rec_annotations = []

#     annotated_dict = {}
#     for s in filenames:
#         a = Extract_Annotation_Data(s, self.root)
#         # rec_annotations.append(a)
#         annotated_dict[s] = a
#     return annotated_dict

# def Extract_Annotation_Data(file_name, root):
#     recording_annotations = pd.read_csv(os.path.join(
#         root, file_name + '.txt'), names=['Start', 'End', 'Crackles', 'Wheezes'], delim_whitespace=True)
#     return recording_annotations