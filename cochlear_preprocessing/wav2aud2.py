import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.python.lib.io import file_io
import pickle
import os
import pandas as pd
from scipy.signal import lfilter
from decimal import *
from generate_dataset import *

print("Tensorflow Version: {}".format(tf.__version__))


""" Function to 

Arguments:

 x -> the acoustic input.
 cochlear parameters ->
 -	frmlen	: frame length, typically, 8, 16 or 2^[natural #] ms.
 -	tc	: time const., typically, 4, 16, or 64 ms, etc.
	 	  if tc == 0, the leaky integration turns to short-term avg.
 -	fac	: nonlinear factor (critical level ratio), typically, .1 for
		  a unit sequence, e.g., X -- N(0, 1);
		  The less the value, the more the compression.
		  fac = 0,  y = (x > 0),   full compression, booleaner.
		  fac = -1, y = max(x, 0), half-wave rectifier
	 	  fac = -2, y = x,         linear function
 - 	shft	: shifted by # of octave, e.g., 0 for 16k, -1 for 8k,
		  etc. SF = 16K * 2^[shft].%
 - FULLT
 - FULLX
 - bp
 coch_b -> 
 coch_a ->
 order ->
 
Returns:
 
 v5 -> wav2aud()
 L_frm -> frame length used
 
"""

def get_acoustic_input():
    root = '/Users/alirachidi/Documents/Sonavi_Labs/ObjectDetection/data/raw_audios/icbhi_preprocessed_v2_8000'
    sr = 8000
    window_length = Decimal('2')
    step_size = Decimal('1')
    threshold = Decimal('{}'.format(0.15))
    filenames = [s.split('.')[0]
                 for s in os.listdir(path=root) if '.wav' in s]
    rec_annotations = []

    rec_annotations_dict = {}
    for s in filenames:
        a = Extract_Annotation_Data(s, root)
        rec_annotations.append(a)
        rec_annotations_dict[s] = a
        
    cycle_list = []
    
    file_id = 0
    
    for file_name in filenames:
        data = get_icbhi_samples(
            rec_annotations_dict[file_name], file_name, root, sr, file_id, threshold, window_length, step_size)
        cycles_with_labels = []
        for d in data[2:]:
            cycles_with_labels.append((d[0], d[1], d[2], file_id))
        cycle_list.extend(cycles_with_labels)
        file_id += 1
        break # TODO: remove
    return cycle_list

def wav2aud2(x, cochlear_parameters, coch_b, coch_a, order):
    # get filter bank
    verb = 0 # verbose mode
    filt = 'p'
    L, M = coch_b.shape
    print("Shape: [{}, {}]".format(L, M))
    L_x = len(x)
    
    # octave shift, nonlinear factor, frame length, leaky integration
    shft = cochlear_parameters["shft"] #octave shift (Matlab index: 4)
    fac = cochlear_parameters["fac"] # nonlinear factor (Matlab index: 3)
    frmlen = cochlear_parameters["frmlen"] # (Matlab index: 1)
    L_frm = np.round(frmlen * (2**(4+shft)))   #frame length (points)
    tc = cochlear_parameters["tc"] # time constant (Matlab index: 2)
    tc = np.float64(tc)
    alph_exponent = -(1/(tc*2**(4+shft)))
    # print(alph_exponent)
    alph = np.exp(alph_exponent) # decaying factor
    
    # inner ear hair cell time constant in ms
    haircell_tc = np.float64(0.5) # inner ear hair cell time constant in ms
    beta_exponent = np.float64(-1/(haircell_tc*2**(4+shft)))
    # print(beta_exponent)
    beta = np.exp(beta_exponent)
    
    # get data, allocate memory for output
    # print(L_x)
    # print(L_frm)
    N = np.ceil(L_x/L_frm) # number of frames
    print("N: {}".format(N))
    # print(x.shape)
    # x = generate_padded_samples(x, ) # TODO: come back to padding
    
    v5 = np.zeros([int(N), M - 1])
    
    ##############################
    # last channel (highest frequency)
    ##############################

    # get filters from stored matrix
    # print(L)
    # print(M)
    p = int(order[0])
    B =  coch_b[2:(p+2), M - 1]
    A =  coch_a[2:(p+2), M - 1]
    # print(B.shape)
    # print(A.shape)
    y1 = lfilter(B, A, x)
    # print(y1)
    # print(y1.shape)
    y2 = y1
    y2_h = y2
    
    # All other channels
    for ch in range(M-2, 0, -1):
        # ANALYSIS: cochlear filterbank
        # IIR: filter bank convolution ---> y1
        p = int(order[ch])
        B =  coch_b[2:(p+2), ch]
        A =  coch_a[2:(p+2), ch]
        y1 = lfilter(B, A, x)
        
        # TRANSDUCTION: hair cells
        y2 = y1    
        
        # REDUCTION: lateral inhibitory network
        # masked by higher (frequency) spatial response
        y3 = y2 - y2_h
        y2_h = y2
        
        # half-wave rectifier
        # print(len(y3))
        # y4 = max(y3, 0)
        y3[y3 < 0] = 0
        
        # temporal integration window
        # leaky integration. alternative could be simpler short term average
        # print(alph)
        # print(len([1-alph]))
        # print(len(y3))
        y5 = lfilter([1], [1-alph], y3)
        # print(ch)
        # print(L_frm)
        v5_row = []
        for i in range(1, int(N) + 1):
            v5_row.append(y5[int(L_frm*i) - 1])
        print("here")
        print(len(v5_row))
        print(len(v5[:, ch]))
        v5[:, ch] = v5_row
        # v5[:, ch] = y5[int(L_frm):int(L_frm*N)] # N elements, each space with L_frm
    return v5
    
    
    
def main():

    coch_path = '/Users/alirachidi/Documents/Sonavi_Labs/ObjectDetection/Cochlear Preprocessing'
    fs = 4000
    bp = 1
    cochlear_parameters = {"frmlen": 8*(1/(fs/8000)), "tc": 8, "fac": -2, "shft": np.log2(fs/16000), "FULLT": 0, "FULLX": 0, "bp": bp}
    wav_list = get_acoustic_input() # acoustic input
    coch_a = np.loadtxt(str(coch_path + '/COCH_A.txt'), delimiter=',')
    coch_b = np.loadtxt(str(coch_path + '/COCH_B.txt'), delimiter=',')
    p = np.loadtxt(str(coch_path + '/p.txt'), delimiter=',')
    # print(coch_a.shape)
    # print(coch_b.shape)
    # print(p.shape)
    for wav in wav_list:
        wav2aud2(wav[0], cochlear_parameters, coch_b, coch_a, p)
        # TODO: **(1/3)
        break
    # TODO: save the spectrograms somwhere 


if __name__ == "__main__":
    main()
