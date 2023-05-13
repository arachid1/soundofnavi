
from ..main import parameters as p
from collections import defaultdict
from ..patient.Patient import Patient
from ..recording.Recording import Recording
from ..slice.Slice import Slice
import os
import librosa
import types
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt

# TODO: helper functions to return patients/recordings/slices
#TODO: change lamda of all objs?
class Dataset:
    '''
    :param root: root folde of the dataset
    '''
    def __init__(self,id, root, metadata_root, get_filenames):
        self.id = id
        self.root = root
        self.metadata_root = metadata_root
        self.get_filenames = types.MethodType(get_filenames, self) 
        self.logs_folder = ''
        self.patients = defaultdict(lambda: None)
        self.testing_files_registry = []
    
    ######### Workflow main functions #########
    
    def load_recordings(self):
        # filenames without .wav
        self.filenames = self.get_filenames() 
        # if testing mode, calls hard-coded filenames for each dataset
        if p.testing:
            self.filenames = self.get_testing_filenames()
        # a dictionary of labels for ALL the filenames
        label_dict = self.build_label_dict()
        for filename in self.filenames:
            patient_id = self.get_patient_id(filename)
            audio = self.load_audio(filename)
            # accessing each of the dictonary for individual labels using patient id 
            label = self.read_recording_label(label_dict, patient_id)
            # the filename must be stripped of its path (relative or absolute) and extension (.wav)
            # NOTE: except for BD?
            if patient_id in self.patients.keys():
                recording = Recording(filename, audio, label, self.patients[patient_id]) 
                self.patients[patient_id].recordings[filename] = recording
            else:
                patient = Patient(patient_id, self)
                recording = Recording(filename, audio, label, patient) 
                patient.recordings[filename] = recording
                self.patients[patient_id] = patient
    
    def prepare_slices(self):
        for patient_id, patient in self.patients.items():
            for recording_id, recording in patient.recordings.items():
                self.slice_recording(recording)
    
    def slice_recording(self, recording):
        audio = recording.audio
        label = recording.label
        filename = recording.id
        rw_index = 0
        max_ind = len(audio)
        additional_step = True
        order = 0
        while additional_step:
            start = rw_index
            end = rw_index + p.audio_length
            audio_c, start_ind, end_ind = self.slice_array(start, end, audio, p.sr, max_ind)
            if (not (abs(end_ind - start_ind) == (p.audio_length * p.sr))):  # ending the loop
                additional_step = False
                if (abs(start_ind - end_ind) < (p.sr * p.audio_length * Decimal(0.5))):  # disregard if LOE than fraction of the audio length (<=1 for 0.5 using 2sec, <=5 for 10sec) 
                    continue
                else:  # 0 pad if more than half of audio length
                    audio_c = self.pad_sample(
                        audio_c, p.sr * p.audio_length)
            # find_label returns an integer (-1 if none)
            dataset_id = recording.patient.dataset.id
            if p.mode == "cw" and (dataset_id == "Icbhi" or dataset_id == "Perch" or dataset_id == "Antwerp"): #TODO: fix
                audio_c_label = self.read_slice_label(label, start_ind, end_ind, )
            else:
                audio_c_label = self.read_slice_label(label)
            audio_c_filename = self.generate_slice_filename(filename, order, additional_step)
            slice = Slice(audio_c_filename, audio_c, audio_c_label, recording)
            recording.slices[audio_c_filename] = slice
            rw_index = start + p.step_size
            order += 1
     
    ######### Loading workflow helpers #########
    def load_audio(self, filename):
        path = os.path.join(self.root, filename + '.wav')
        data, ___ = librosa.load(path, sr=p.sr)
        return data
    
    def get_filenames(self): # overwritten
        pass
    
    def get_testing_filenames(self): # overwritten
        pass
        # TODO: write a function that reads a files of custom testing files from testing_file_registry
        
    def build_label_dict(self): # overwritten
        pass
    
    def read_recording_label(self): # overwritten
        pass
    
    def get_patient_id(self): # overwritten
        pass
    
    ######### Preparing workflow helpers #########
    def generate_slice_filename(self, filename, order, additional_step):
        '''
        generates filename for chunk by adding '_i' with i = 0-indexed order of the chunk. 
        If last chunk,  i == 99
        '''
        if not additional_step:
            order = 99
        return filename + '_' + str(order)

    def slice_array(self, start, end, raw_data, sample_rate, max_ind=10e+5):
        '''
        slices the [start, end] chunk of raw_data
        '''
        start_ind = min(int(start * sample_rate), max_ind)
        end_ind = min(int(end * sample_rate), max_ind)
        return raw_data[start_ind: end_ind], start_ind, end_ind
    
    def pad_sample(self, source, output_length):
        '''
        pad source to output_length
        '''
        output_length = int(output_length)
        copy = np.zeros(output_length, dtype=np.float32)
        src_length = len(source)
        frac = src_length / output_length
        if(frac < 0.5):
            # tile forward sounds to fill empty space
            cursor = 0
            while(cursor + src_length) < output_length:
                copy[cursor:(cursor + src_length)] = source[:]
                cursor += src_length
        else:
            copy[:src_length] = source[:]
        return copy
    
    def read_slice_label(self, recording_dict): # overwritten
        pass
    
    ######### Return helper functions #########
    def return_recordings_by_patient(self):
        d = {}
        for patient_id, patient in self.patients.items():
            d[patient_id] = patient.recordings.values()
            # d[p.id] = p.recordings
        return d
    
    def return_slices_by_patient(self):
        d = defaultdict(lambda: [])
        for patient_id, patient in self.patients.items():
            for recording_id, recording in patient.recordings.items():
                d[patient_id].append(recording.slices.values())
        return d
    
    def return_slices_by_recording_by_patient(self):
        d = defaultdict(lambda: {})
        for patient_id, patient in self.patients.items():
            d[patient_id] = patient.return_slices_by_recording()
        return d
        # uses return_slices_by_measurement in Patient(), which uses return_slices() in Measurement()
        
    ######### Analysis helper functions #########
    
    def plot(self, data, nrows, ncols):
        plt.figure(figsize=(15, 12))
        plt.subplots_adjust(hspace=0.2)
        # plt.suptitle("Daily closing prices", fontsize=18, y=0.95)

        # set number of columns (use 3 to demonstrate the change)
        ncols = 3
        # calculate number of rows
        nrows = len(data) // ncols + (len(data) % ncols > 0)

        # loop through the length of tickers and keep track of index
        for n, el in enumerate(data):
            # add a new subplot iteratively using nrows and cols
            ax = plt.subplot(nrows, ncols, n + 1)

            # filter df and plot ticker on the new subplot axis
            to_plot = func(el)
            to_plot.plot(ax=ax)

            # chart formatting
            ax.set_title(el.upper())
            ax.get_legend().remove()
            ax.set_xlabel("")
        # fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
        # fig.suptitle(self.id)
        # pass
    
    def get_dataset_profile():
        pass
    
    def get_patient_metadata(self): # overwritten
        pass