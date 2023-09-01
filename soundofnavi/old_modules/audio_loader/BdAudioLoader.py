from .AudioLoader import AudioLoader
from ..main import parameters
import pandas as pd
import os

class BdAudioLoader(AudioLoader):

    def __init__(self, root, get_filenames, excel_path):
        super().__init__(root, get_filenames)
        self.root = root 
        self.label_root = root
        self.excel_path = excel_path
        self.name = "Bd"

    # example: /.../.../0270792_170803_001-1_F -> 0270792_170803_001-1_F -> int(0270792) -> 270792
    def get_patient_id(self, filename):
        filename = filename.split('/')[-1]
        s = filename.split('_')[0]
        s = filename.split('-')[0]
        return int(s)

    # reading the label indicated by column PEP1, which is 1 or 0
    def read_label(self, _dict, patient_id):
        file_column = _dict.loc[_dict['HOSP_ID'] == patient_id]
        if file_column.empty:
            print("Excel column for patient {} is empty".format(patient_id))
            return -1
        label = str(file_column['PEP1'].values[0])
        return label
    
    # opening Bangladesh_PCV_onlyStudyPatients.xlsx 
    def build_label_dict(self): #TODO: standarize to reutnr the same output across objects, maybe an array so it's faster access
        label_dict = pd.read_excel(self.excel_path, engine='openpyxl')
        return label_dict
    
    def get_testing_filenames(self):
        return ['../../data/PCV_SEGMENTED_Processed_Files/0270792_SEGMENTED/0270792_170803_001-1_F',  '../../data/PCV_SEGMENTED_Processed_Files/0270792_SEGMENTED/0270792_170803_001-2_F',
        '../../data/PCV_SEGMENTED_Processed_Files/0270790_SEGMENTED/0270790_170802_005-1_F',  '../../data/PCV_SEGMENTED_Processed_Files/0270790_SEGMENTED/0270790_170802_005-2_F',
        '../../data/PCV_SEGMENTED_Processed_Files/0270814_SEGMENTED/0270814_170803_005-1_F',  '../../data/PCV_SEGMENTED_Processed_Files/0270814_SEGMENTED/0270814_170803_005-2_F',
        '../../data/PCV_SEGMENTED_Processed_Files/0270842_SEGMENTED/0270842-170805-000-1_F', '../../data/PCV_SEGMENTED_Processed_Files/0270842_SEGMENTED/0270842-170805-000-2_F'
        ]
        # return ['0270792_SEGMENTED/0270792_170803_001-1_F',  '0270792_SEGMENTED/0270792_170803_001-2_F',
        # '0270792_SEGMENTED/0270790_170802_005-1_F',  '0270792_SEGMENTED/0270790_170802_005-2_F',
        # '0270814_170803_005-1_F',  '0270792_SEGMENTED/0270814_170803_005-2_F',
        # '0270842-170805-000-1_F', '0270842-170805-000-2_F']
