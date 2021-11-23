import types
from ..main import parameters
import numpy as np
from decimal import Decimal

""" Audio Preparer Object
This object overall converts full length recordings into chunks of fixed length, with defined slicing, overlap and labeling parameters/functions.
It can return data grouped by patient (default), chunk or full length recording based on parameters.

This is where most things happen, and in contrast to AudioLoader, it has less data-specific functions. However, each dataset-specific preparer has to implement 
- find_labels()
- return_all_samples_by_patient()

EXAMPLE:

self.samples = {'Jordan': [
    [array([-2.083e-06, -9.467e-07,  2.297e-06, ...,  2.025e-02,  1.845e-02, 9.768e-03], dtype=float32), 'Asthma', 'BP2_Asthma,E W,P L L R,52,F'], 
    [array([-5.110e-07, -5.519e-07,  5.078e-07, ..., -2.021e-02, -1.784e-02,-9.300e-03], dtype=float32), 'Plueral Effusion', 'BP6_Plueral Effusion,I C B,P L R,81,M'], 
    
    [array([ 9.815e-08,  9.122e-08, -1.099e-07, ...,  9.202e-03,  7.932e-03,
    4.084e-03], dtype=float32), 'heart failure', 'BP19_heart failure,C,P R U,70,F'], [array([-5.275e-07,  2.542e-07,  6.650e-07, ..., -3.311e-01, -2.833e-01,
    -1.451e-01], dtype=float32), 'Lung Fibrosis', 'BP26_Lung Fibrosis,Crep,P,90,F'], [array([ 1.514e-06,  3.883e-06, -1.697e-06, ..., -2.258e-02, -2.018e-02,
    -1.059e-02], dtype=float32), 'heart failure', 'BP66_heart failure,Crep,P R L ,43,M'], [array([-5.286e-07, -9.805e-07,  5.038e-07, ...,  3.195e-01,  2.741e-01,
    1.406e-01], dtype=float32), 'COPD', 'BP108_COPD,E W,P R L ,63,M'], [array([ 1.834e-06,  1.748e-06, -1.910e-06, ..., -1.992e-02, -1.821e-02,
    -9.656e-03], dtype=float32), 'N', 'BP100_N,N,P R M,70,F'], [array([-8.385e-07, -1.531e-06,  8.435e-07, ..., -7.499e-02, -6.563e-02,
    -3.405e-02], dtype=float32), 'pneumonia', 'BP18_pneumonia,C,P R U,57,M'], [array([ 1.865e-07, -2.514e-07, -2.634e-07, ...,  1.628e-01,  1.399e-01,
    7.185e-02], dtype=float32), 'N', 'BP87_N,N,P R M,72,M'], [array([ 8.769e-07,  8.632e-07, -9.404e-07, ...,  8.454e-02,  7.326e-02,
    3.781e-02], dtype=float32), 'pneumonia', 'BP37_pneumonia,Crep,A R L,70,F']]}

self.chunked_samples = 
{'Jordan': [
    [
        [array([-2.083e-06, -9.467e-07,  2.297e-06, ..., -2.410e-01, -2.418e-01, 2.427e-01], dtype=float32), 0, 'BP2_Asthma,E W,P L L R,52,F_0'], [array([ 0.108,  0.107,  0.106, ..., -0.039, -0.037, -0.035], dtype=float32), 0, 'BP2_Asthma,E W,P L L R,52,F_1'], [array([-0.243, -0.244, -0.245, ..., -0.017, -0.017, -0.017], dtype=float32), 0, 'BP2_Asthma,E W,P L L R,52,F_2'], [array([-0.033, -0.031, -0.028, ..., -0.09 , -0.093, -0.096], dtype=float32), 0, 'BP2_Asthma,E W,P L L R,52,F_3'], [array([-0.018, -0.018, -0.019, ...,  0.02 ,  0.018,  0.01 ], dtype=float32), 0, 'BP2_Asthma,E W,P L L R,52,F_99']
    ],
    [
        [array([-5.110e-07, -5.519e-07,  5.078e-07, ..., -4.726e-02, -4.730e-02, -4.735e-02], dtype=float32), 0, 'BP6_Plueral Effusion,I C B,P L R,81,M_0'], [array([0.029, 0.03 , 0.03 , ..., 0.065, 0.065, 0.064], dtype=float32), 0, 'BP6_Plueral Effusion,I C B,P L R,81,M_1'], [array([-0.047, -0.047, -0.048, ...,  0.038,  0.038,  0.039], dtype=float32), 0, 'BP6_Plueral Effusion,I C B,P L R,81,M_2'], [array([ 0.064,  0.064,  0.064, ..., -0.008, -0.008, -0.008], dtype=float32), 0, 'BP6_Plueral Effusion,I C B,P L R,81,M_3'], [array([0.039, 0.039, 0.039, ..., 0.   , 0.   , 0.   ], dtype=float32), 0, 'BP6_Plueral Effusion,I C B,P L R,81,M_99']
    ], 
    [[array([-8.385e-07, -1.531e-06,  8.435e-07, ...,  1.693e-02,  1.764e-02,
    1.835e-02], dtype=float32), 1, 'BP18_pneumonia,C,P R U,57,M_0'], [array([-0.027, -0.028, -0.028, ..., -0.009, -0.009, -0.009], dtype=float32), 1, 'BP18_pneumonia,C,P R U,57,M_1'], [array([ 0.019,  0.02 ,  0.02 , ..., -0.002, -0.002, -0.003], dtype=float32), 1, 'BP18_pneumonia,C,P R U,57,M_2'], [array([-0.009, -0.008, -0.008, ...,  0.   ,  0.   ,  0.   ], dtype=float32), 1, 'BP18_pneumonia,C,P R U,57,M_99']], [[array([ 9.815e-08,  9.122e-08, -1.099e-07, ..., -3.031e-02, -3.024e-02,
    -3.018e-02], dtype=float32), 0, 'BP19_heart failure,C,P R U,70,F_0'], [array([-0.028, -0.028, -0.029, ...,  0.002,  0.002,  0.002], dtype=float32), 0, 'BP19_heart failure,C,P R U,70,F_1'], [array([-0.03 , -0.03 , -0.03 , ...,  0.005,  0.005,  0.004], dtype=float32), 0, 'BP19_heart failure,C,P R U,70,F_2'], [array([0.002, 0.002, 0.002, ..., 0.   , 0.   , 0.   ], dtype=float32), 0, 'BP19_heart failure,C,P R U,70,F_99']], [[array([-5.275e-07,  2.542e-07,  6.650e-07, ..., -3.806e-01, -3.810e-01,
    -3.811e-01], dtype=float32), 0, 'BP26_Lung Fibrosis,Crep,P,90,F_0'], [array([0.034, 0.035, 0.037, ..., 0.304, 0.305, 0.305], dtype=float32), 0, 'BP26_Lung Fibrosis,Crep,P,90,F_1'], [array([-0.381, -0.381, -0.38 , ...,  0.   ,  0.   ,  0.   ], dtype=float32), 0, 'BP26_Lung Fibrosis,Crep,P,90,F_99']], [[array([ 8.769e-07,  8.632e-07, -9.404e-07, ...,  1.318e-01,  1.329e-01,
    1.338e-01], dtype=float32), 1, 'BP37_pneumonia,Crep,A R L,70,F_0'], [array([0.196, 0.194, 0.192, ..., 0.   , 0.   , 0.   ], dtype=float32), 1, 'BP37_pneumonia,Crep,A R L,70,F_99']], [[array([ 1.514e-06,  3.883e-06, -1.697e-06, ..., -2.238e-02, -2.280e-02,
    -2.322e-02], dtype=float32), 0, 'BP66_heart failure,Crep,P R L ,43,M_0'], [array([-0.012, -0.011, -0.009, ...,  0.013,  0.013,  0.013], dtype=float32), 0, 'BP66_heart failure,Crep,P R L ,43,M_1'], [array([-0.024, -0.024, -0.024, ...,  0.   ,  0.   ,  0.   ], dtype=float32), 0, 'BP66_heart failure,Crep,P R L ,43,M_99']], [[array([ 1.865e-07, -2.514e-07, -2.634e-07, ...,  1.018e-01,  1.016e-01,
    1.014e-01], dtype=float32), 0, 'BP87_N,N,P R M,72,M_0'], [array([-0.002, -0.003, -0.003, ...,  0.108,  0.109,  0.109], dtype=float32), 0, 'BP87_N,N,P R M,72,M_1'], [array([0.101, 0.101, 0.1  , ..., 0.   , 0.   , 0.   ], dtype=float32), 0, 'BP87_N,N,P R M,72,M_99']], [[array([ 1.834e-06,  1.748e-06, -1.910e-06, ..., -1.191e-01, -1.226e-01,
    -1.259e-01], dtype=float32), 0, 'BP100_N,N,P R M,70,F_0'], [array([-0.246, -0.248, -0.249, ...,  0.   ,  0.   ,  0.   ], dtype=float32), 0, 'BP100_N,N,P R M,70,F_99']], [[array([-5.286e-07, -9.805e-07,  5.038e-07, ..., -1.631e-01, -1.639e-01,
    -1.646e-01], dtype=float32), 0, 'BP108_COPD,E W,P R L ,63,M_0'], [array([-0.228, -0.228, -0.228, ...,  0.049,  0.038,  0.028], dtype=float32), 0, 'BP108_COPD,E W,P R L ,63,M_1'], [array([-0.165, -0.166, -0.167, ...,  0.211,  0.219,  0.228], dtype=float32), 0, 'BP108_COPD,E W,P R L ,63,M_2'], [array([ 0.018,  0.008, -0.002, ...,  0.   ,  0.   ,  0.   ], dtype=float32), 0, 'BP108_COPD,E W,P R L ,63,M_99']]]
}

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
                print("Warning: it seems the sample {} was less than 5 seconds (Refer to AudioPreparer)".format(sample))
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