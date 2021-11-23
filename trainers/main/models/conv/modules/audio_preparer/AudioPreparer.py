import types
from ..main import parameters
import numpy as np
from decimal import Decimal

""" Audio Preparer Object
*** This is where most things happen, but each dataset-specific preparer has to implement 
- find_labels
- return_all_samples_by_patient
This object overall converts full length recordings into N chunks of fixed length with appropriate labeling.
can return data grouped by chunk, full length recording or by patient.
"""
class AudioPrepaper:
    
    def __init__(self, samples, name, mode):
        self.samples = samples
        self.name = name
        self.mode = mode
        self.chunked_samples = []
    
    def prepare_all_samples(self):
        '''
        applies 'prepare_singular_sample' to every sample and ass
        input: full length samples -> list
        output: chunked samples -> list of lists
        '''
        for sample in self.samples:
            chunked_sample = self.prepare_singular_sample(sample)
            if len(chunked_sample) == 0:
                print("It seems the sample {} was less than 5 seconds (Refer to AudioPreparer)".format(sample))
                continue
            self.chunked_samples.append(chunked_sample)

    def prepare_singular_sample(self, sample):
        '''
        function that decomposes a full length sample into its individual elements, and ensures proper labelling and filenaming
        for example, a Jordan sample for pneumonia will return a list of 
        input: full length sample -> list [full length audio, label, filename]
        output: chunked_sample -> list of lists [audio chunk #1, label of chunk #1, name of chunk #1], ..., [audio chunk #n, label of chunk #n, name of chunk #n]]
        '''
        chunked_sample = []
        audio = sample[0]
        label = sample[1]
        filename = sample[2]
        rw_index = 0
        max_ind = len(audio)
        additional_step = True
        order = 0
        while additional_step:
            start = rw_index
            end = rw_index + parameters.audio_length
            audio_c, start_ind, end_ind = self.slice_array(start, end, audio, parameters.sr, max_ind)
            if (not (abs(end_ind - start_ind) == (parameters.audio_length * parameters.sr))):  # ending the loop
                additional_step = False
                if (abs(start_ind - end_ind) <= (parameters.sr * parameters.audio_length * Decimal(0.5))):  # disregard if LOE than half of the audio length (<=1 for 2sec, <=5 for 10sec) 
                    continue
                else:  # 0 pad if more than half of audio length
                    audio_c = self.pad_sample(
                        audio_c, parameters.sr * parameters.audio_length)
            # find_label returns an integer (-1 if none)
            if self.mode == "cw" and (self.name == "Icbhi" or self.name == "Perch" or self.name == "Antwerp"):
                audio_c_label = self.find_label(label, start_ind, end_ind, )
            else:
                audio_c_label = self.find_label(label)
            audio_c_filename = self.generate_chunk_filename(filename, order, additional_step)
            chunked_sample.append([audio_c, audio_c_label, audio_c_filename])
            rw_index = start + parameters.step_size
            order += 1

        if len(chunked_sample) > 0:
            chunked_sample[-1][2] = self.generate_chunk_filename(filename, 99, additional_step)

        return chunked_sample

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

    def find_label(self, recording_dict):
        pass

    def slice_array(self, start, end, raw_data, sample_rate, max_ind=10e+5):
        '''
        slices the [start, end] chunk of raw_data
        '''
        start_ind = min(int(start * sample_rate), max_ind)
        end_ind = min(int(end * sample_rate), max_ind)
        return raw_data[start_ind: end_ind], start_ind, end_ind
    
    def generate_chunk_filename(self, filename, order, additional_step):
        '''
        generates filename for chunk by adding '_i' with i = 0-indexed order of the chunk. 
        If last chunk,  i == 99
        '''
        if not additional_step:
            order = 99
        return filename + '_' + str(order)
    
    def return_all_samples(self):
        return self.chunked_samples
    
    def return_patient_id(self, filename):
        return None

    def return_all_samples_by_patient(self):
        
        samples_by_patients = {}
        for s in self.chunked_samples:
            patient_id = self.return_patient_id(s[0][2])
            if patient_id == "None":
                print("patient_id is none inside return_all_Samples")
                exit()
            if patient_id not in samples_by_patients:
                samples_by_patients[patient_id] = s
            else:
                samples_by_patients[patient_id] += s
        samples_by_patients = list(samples_by_patients.values())
        self.chunked_samples = samples_by_patients
        return self.chunked_samples 