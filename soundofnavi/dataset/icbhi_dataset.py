from soundofnavi.main import parameters as p
from soundofnavi.patient.patient import Patient
from soundofnavi.recording.recording import Recording
from soundofnavi.dataset.dataset import Dataset
from collections import defaultdict
import os
import librosa
import types
import pandas as pd
from decimal import Decimal


class IcbhiDataset(Dataset):
    """
    :param root: root folde of the dataset
    """

    def __init__(self, root, id, metadata_path, get_filenames):
        super().__init__(root, id, metadata_path, get_filenames)
        self.name = "Icbhi"
        if p.mode == "cw":
            self.build_label_dict = types.MethodType(build_cw_label_dict, self)
            self.read_recording_label = types.MethodType(read_cw_recording_label, self)
            self.read_slice_label = types.MethodType(read_cw_slice_label, self)
        else:
            self.build_label_dict = types.MethodType(build_pneumonia_label_dict, self)
            self.read_recording_label = types.MethodType(
                read_pneumonia_recording_label, self
            )
            self.read_slice_label = types.MethodType(read_pneumonia_slice_label, self)

    ######## Loading workflow helpers #########
    def get_patient_id(self, filename):

        if p.mode == "cw":
            return filename
        elif p.mode == "pneumonia":
            return int(filename[:3])

    def build_metadata(self):
        return pd.read_csv(
            self.metadata_path,
            names=[
                "Patient_id",
                "Age",
                "Sex",
                "Adult BMI (kg/m2)",
                "Child Weight (kg)",
                "Child Height (cm)",
            ],
            delimiter=" ",
        )
        # return df_no_diagnosis[df_no_diagnosis['Patient_id'] == int(patient_id)]

    # def get_patient_metadata(self, patient_id):
    #     df_no_diagnosis = pd.read_csv(self.metadata_path, names =
    #     ['Patient_id', 'Age', 'Sex' , 'Adult BMI (kg/m2)', 'Child Weight (kg)' , 'Child Height (cm)'],
    #     delimiter = ' ')
    #     return df_no_diagnosis[df_no_diagnosis['Patient_id'] == int(patient_id)]

    def get_filenames(self):
        pass

    ######## Preparing workflow helpers #########


######## Loading workflow helpers (2) #########
def read_cw_recording_label(self, _dict, patient_id):
    return _dict[patient_id]


def read_pneumonia_recording_label(self, _dict, patient_id):
    return _dict.loc[patient_id]["Diagnosis"]


def build_pneumonia_label_dict(self):
    label_dict = pd.read_csv(
        os.path.join(self.root, "patient_diagnosis.csv"),
        names=["Diagnosis"],
        delimiter=",",
        index_col=0,
    )
    return label_dict


def build_cw_label_dict(self):
    annotated_dict = {}
    for s in self.filenames:
        a = read_cw_annotation(s, self.root)
        annotated_dict[s] = a
    return annotated_dict


def read_cw_annotation(file_name, root):
    recording_annotations = pd.read_csv(
        os.path.join(root, file_name + ".txt"),
        names=["Start", "End", "Crackles", "Wheezes"],
        delim_whitespace=True,
    )
    return recording_annotations


######### Preparing workflow helpers (2) #########
# TODO: move this somewhere else
def read_pneumonia_slice_label(self, recording_dict):
    #  as obtained from patient_diagnosis.csv
    return 1 if recording_dict == "Pneumonia" else 0
    # print(recording_dict)
    # if recording_dict['Diagnosis'] == 'Pneumonia':
    #     return 1
    # else:
    #     return 0


def read_cw_slice_label(self, recording_dict, start, end):
    times, ___ = find_cw_times(recording_dict, p.sr, True)
    l = find_cw_labels(times, start, end, p.overlap_threshold, p.audio_length, p.sr)
    return l


def find_cw_times(recording_annotations, sample_rate, first):
    times = []
    for i in range(len(recording_annotations.index)):
        row = recording_annotations.loc[i]
        c_start = Decimal("{}".format(row["Start"]))
        c_end = Decimal("{}".format(row["End"]))
        if c_end - c_start < 1:
            continue
        if first:
            rw_index = c_start
            first = False
        times.append(
            (
                c_start * sample_rate,
                c_end * sample_rate,
                row["Crackles"],
                row["Wheezes"],
            )
        )  # get all the times w labels
    return times, rw_index


def find_cw_labels(times, start, end, overlap_threshold, audio_length, sample_rate):

    overlaps = []
    for time in times:
        if start > time[0]:  # the window starts after
            if start >= time[1]:  # no overlap
                continue
            if end <= time[1]:  # perfect overlap: window inside breathing pattern
                # shouldn't happen since window is 2 sec long
                if (time[1] - time[0]) < (
                    overlap_threshold * audio_length * sample_rate
                ):
                    continue
                else:
                    return [time[2], time[3]]
            else:  # time[1] < end: partial overlap, look at next window c)
                if (time[1] - start) < (overlap_threshold * audio_length * sample_rate):
                    continue
                else:
                    overlaps.append(time)
        else:  # the window starts before
            if end <= time[0]:  # no overlap
                continue
            if end < time[1]:  # partial overlap b)
                if (end - time[0]) < (overlap_threshold * audio_length * sample_rate):
                    continue
                else:
                    overlaps.append(time)
            else:  # end > time[1]: perfect overlap: breathing pattern inside of window
                # shouldn't happen since all windows are 1 sec or more
                if (time[1] - time[0]) < (
                    overlap_threshold * audio_length * sample_rate
                ):
                    continue
                else:
                    overlaps.append(time)
    return list(process_cw_overlaps(overlaps))


def process_cw_overlaps(overlaps):
    crackles = sum([overlap[2] for overlap in overlaps])
    wheezes = sum([overlap[3] for overlap in overlaps])
    crackles = 1.0 if (not (crackles == 0)) else 0.0
    wheezes = 1.0 if (not (wheezes == 0)) else 0.0
    return crackles, wheezes
