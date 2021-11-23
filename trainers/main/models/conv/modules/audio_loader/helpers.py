import os
import glob
import pandas as pd

def default_get_filenames(self):

    return [s.split('.')[0] for s in os.listdir(path=self.root) 
                        if '.wav' in s]

def perch_get_filenames(self):
    return [s.split('.')[0] + '.' + s.split('.')[1]
                 for s in os.listdir(path=self.root) if 'F.wav' in s]

def bd_get_filenames(self):
    filenames = []
    df = pd.read_excel(self.excel_path, engine='openpyxl')
    folder_paths = glob.glob(os.path.join(self.root, "*"),recursive=True)
    folder_paths = sorted(folder_paths)

    count_1 = 0
    count_2 = 0
    count_3 = 0

    for folder_path in folder_paths: # example: 0194930_SEGMENTED 
        # folders with issues
        if folder_path == os.path.join(self.root, "0365993_SEGMENTED") or folder_path == os.path.join(self.root, "0273320_SEGMENTED") or folder_path == os.path.join(self.root, "0364772_SEGMENTED"):
            continue
        recordings = glob.glob(os.path.join(folder_path, "*.wav")) # example: 0194930_SEGMENTED_1.wav
        recordings = sorted(recordings)
        # folders that don't have enough recordings
        # if len(recordings) != 6:
        #     count_1 += 1
        #     continue
        patient_id = int(folder_path.split('/')[-1].split('_')[0])
        file_column = df.loc[df['HOSP_ID'] == patient_id]
        if file_column.empty:
            count_2 += 1
            continue
        label = str(file_column['PEP1'].values[0])
        if label == "Uninterpretable":
            count_3 += 1
            continue
        # for r in recordings:
        #     f = os.path.splitext(r)[0]
        #     f = os.path.split('/')[-2] + os.path.split('/')[-1]
        #     print(f)
        #     filenames.extend(f)
        # exit()
    print("BD: there are 3 folders with their own issues, {} with empty excel columns and {} with uninterpretable PEP1 value. ".format(count_2, count_3))
    return filenames
