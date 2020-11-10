import numpy as np
# import pickle
import os
from scipy.signal import lfilter
from matplotlib import pyplot as plt
import librosa
from librosa import display
from decimal import *
import wave
import resampy as rs
import scipy.io.wavfile as wf
from math import pi

## ian commenting out this function since it already exists in resampy
## def resample(current_rate, data, target_rate):
##    x_original = np.linspace(0, 100, len(data))
##    new_length = int(len(data) * (target_rate / current_rate))
##    x_resampled = np.linspace(0, 100, new_length)
##    resampled = np.interp(x_resampled, x_original, data)
##    return (target_rate, resampled.astype(np.float32))

def read_wav_file(str_filename, target_rate):
    wav = wave.open(str_filename, mode='r')

    (sample_rate, data) = extract2FloatArr(wav, str_filename)

    if (sample_rate != target_rate):
        # print("is resampling...")
        ## ian commenting out below because of resampy syntax
        #(_, data) = resample(sample_rate, data, target_rate)
        data = rs.resample(data,sample_rate, target_rate)
    return (target_rate, data.astype(np.float32))


def extract2FloatArr(lp_wave, str_filename):
    (bps, channels) = bitrate_channels(lp_wave)

    if bps in [1, 2, 4]:  # depth
        (rate, data) = wf.read(str_filename)
        divisor_dict = {1: 255, 2: 32768}
        if bps in [1, 2]:
            divisor = divisor_dict[bps]
            data = np.divide(data, float(divisor))  # clamp to [0.0,1.0]
        return (rate, data)

    elif bps == 3:
        # 24bpp wave
        return read24bitwave(lp_wave)

    else:
        raise Exception(
            'Unrecognized wave format: {} bytes per sample'.format(bps))


def read24bitwave(lp_wave):
    nFrames = lp_wave.getnframes()
    buf = lp_wave.readframes(nFrames)
    reshaped = np.frombuffer(buf, np.int8).reshape(nFrames, -1)
    short_output = np.empty((nFrames, 2), dtype=np.int8)
    short_output[:, :] = reshaped[:, -2:]
    short_output = short_output.view(np.int16)
    # return numpy array to save memory via array slicing
    return (lp_wave.getframerate(), np.divide(short_output, 32768).reshape(-1))


def bitrate_channels(lp_wave):
    bps = (lp_wave.getsampwidth() / lp_wave.getnchannels())  # bytes per sample
    return (bps, lp_wave.getnchannels())

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

def write_data(array, name):
    with open(name,'w') as f:
            for val in array:
                f.write(str(val) + ", ")
            f.write('\n')
    

def wav2aud2(x, cochlear_parameters, coch_b, coch_a, order, write_folder):
    print("Start...")
    # get filter bank
    ## both of the following lines are not used so I removed them
    # verb = 0 # verbose mode
    # filt = 'p' 
    L, M = coch_b.shape
    print("Shape: [{}, {}]".format(L, M))
    L_x = len(x)

    Fs = 8000
    
    # octave shift, nonlinear factor, frame length, leaky integration
    shft = cochlear_parameters["shft"] #octave shift (Matlab index: 4)
    fac = cochlear_parameters["fac"] # nonlinear factor (Matlab index: 3)
    frmlen = cochlear_parameters["frmlen"] # (Matlab index: 1)
    L_frm = np.round(frmlen * (2**(4+shft)))   #frame length (points)
    tc = cochlear_parameters["tc"] # time constant (Matlab index: 2)
    tc = np.float64(tc)
    alph_exponent = -(1/(tc*(2**(4+shft))))
    alph = np.exp(alph_exponent) # decaying factor
    
    # inner ear hair cell time constant in ms
    # haircell_tc = np.float64(0.5) # inner ear hair cell time constant in ms
    # beta_exponent = np.float64(-1/(haircell_tc*(2**(4+shft))))
    # beta = np.exp(beta_exponent)
    
    # get data, allocate memory for output
    N = np.ceil(L_x/L_frm) # number of frames
    print("Number of frames: {}".format(N))
    # x = generate_padded_samples(x, ) # TODO: come back to padding
    
    v5 = np.zeros([int(N), M - 1])
    
    ##############################
    # last channel (highest frequency)
    ############################## 

    # get filters from stored matrix
    print("Number of filters: {}".format(M))
    #p = int(order[0]) # this order is wrong
    p = int(order[M-1]) # ian's change 11/6

    # ian changed to 0 because of the way p, coch_a, and coch_b are seperated
    B =  coch_b[0:p+1, M - 1] # M-1 before
    A =  coch_a[0:p+1, M - 1]
    y1 = lfilter(B, A, x)
    y2 = y1
    y2_h = y2
    
    column_folder = write_folder + '/' + 'column' + str(M) + '/'
    os.makedirs(column_folder, exist_ok=True)
    
    write_data(y1, column_folder + "arma_filter.txt")
        
    # All other channels
    for ch in range(M-2, -1, -1):
        column_folder = write_folder + '/' + 'column' + str(ch+1) + '/'
        os.makedirs(column_folder, exist_ok=True)
        # ANALYSIS: cochlear filterbank
        # IIR: filter bank convolution ---> y1  
        p = int(order[ch])
        B =  coch_b[0:p+1, ch]
        A =  coch_a[0:p+1, ch]

        y1 = lfilter(B, A, x)

        # TRANSDUCTION: hair cells
        y2 = y1   
        
        write_data(y2, column_folder + "arma_filter.txt")
        
        # REDUCTION: lateral inhibitory network 
        # masked by higher (frequency) spatial response
        y3 = y2 - y2_h
        y2_h = y2
        
        write_data(y3, column_folder + "lat_inhib_mask.txt")
        
        # half-wave rectifier 
        y4 = y3.copy()
        y4[y4 < 0] = 0
            
        write_data(y4, column_folder + "half_wave_rec.txt")
            
        
        # temporal integration window #
        # leaky integration. alternative could be simpler short term average
        y5 = lfilter([1], [1-alph], y4)
        
        write_data(y5, column_folder + "temp_integ_window.txt")
            
        v5_row = []
        for i in range(1, int(N) + 1):
            v5_row.append(y5[int(L_frm*i) - 1])
        v5[:, ch] = v5_row
        # v5[:, ch] = y5[int(L_frm):int(L_frm*N)] # N elements, each space with L_frm
    print("...End")
    
    return v5
    
    
def main():

    #path_common = 'C:/Sonavi Labs/classification_algorithm/'
    path_common = os.getcwd() + '/'
    # coch_path = path_common + 'cochlear_preprocessing/'
    # file_path = path_common + '/data/Normal/K06111-07_F_6.72214725_1_0sec.wav' 
    # Ali 
    coch_path = path_common + 'cochlear_preprocessing/'
    file_path = path_common + 'data/cochlear_processing_validation_data/Normal/K06111-07_F_6.72214725_1_0sec.wav' 
    
    file_name = file_path.split('/')
    file_name = file_name[len(file_name) -1]
    validation_folder = coch_path + "validation/"
    os.makedirs(validation_folder, exist_ok=True)
    
    write_folder = str(validation_folder + file_name)
    # os.makedirs(column_folder, exist_ok=True)

    fs = 8000 #changed by ian 11/6 from a mistake in runme.txt
    bp = 1
    cochlear_parameters = {"frmlen": 8*(1/(fs/8000)), "tc": 8, "fac": -2, "shft": np.log2(fs/16000), "FULLT": 0, "FULLX": 0, "bp": bp}
    # coch_a = np.loadtxt(str(coch_path + '/COCH_A.txt'), delimiter=',')
    # coch_b = np.loadtxt(str(coch_path + '/COCH_B.txt'), delimiter=',')
    # p = np.loadtxt(str(coch_path + '/p.txt'), delimiter=',')
    coch_a = np.loadtxt(coch_path + 'COCH_A.txt', delimiter=',')
    coch_b = np.loadtxt(coch_path + 'COCH_B.txt', delimiter=',')
    p = np.loadtxt(coch_path + 'p.txt', delimiter=',')
    
    sr = 8000
    wav = read_wav_file(file_path, sr)
    wav = wav[1]
    
    spect = wav2aud2(wav, cochlear_parameters, coch_b, coch_a, p, write_folder)
    
    spect = spect**(1/3)
    
    os.makedirs(write_folder, exist_ok=True)

    with open(write_folder + "/" + "full_aud_spect.txt",'w') as f:
        for i in range(spect.shape[0]):
            for y in range(len(spect[i])):
                f.write(str(spect[i][y]) + ", ")
            f.write('\n')
            
    fig = plt.figure(figsize=(20, 10))
    display.specshow(
        spect,
        # y_axis="log",
        sr=sr,
        cmap="coolwarm"
    )
    plt.colorbar()
    plt.show()
    fig.savefig(write_folder + "/" + 'coch_spectrogram.png')
    
    
    
    # # wav_list = get_acoustic_input(root) # #TODO: write to scalle
    # specs = []
    # for wav in wav_list: #TODO: Scale 
    #     spect = wav2aud2(wav[0], cochlear_parameters, coch_b, coch_a, p)
    #     fig = plt.figure(figsize=(20, 10))
    #     # spect = self.X[index][:, :, 0]
    #     display.specshow(
    #         spect,
    #         y_axis="log",
    #         sr=8000,
    #         cmap="coolwarm"
    #     )
    #     plt.colorbar()
    #     plt.show()
    #     # TODO: **(1/3)
    #     break
    # TODO: save the spectrograms somwhere 
    
    # def get_acoustic_input(root):
#     sr = 8000
#     window_length = Decimal('2')
#     step_size = Decimal('1')
#     threshold = Decimal('{}'.format(0.15))
#     filenames = [s.split('.')[0]
#                  for s in os.listdir(path=root) if '.wav' in s]
    
#     rec_annotations = []

#     rec_annotations_dict = {}
#     for s in filenames:
#         a = Extract_Annotation_Data(s, root)
#         rec_annotations.append(a)
#         rec_annotations_dict[s] = a
        
#     cycle_list = []
 
#     file_id = 0
    
#     for file_name in filenames:
#         data = get_icbhi_samples(
#             rec_annotations_dict[file_name], file_name, root, sr, file_id, threshold, window_length, step_size)
#         cycles_with_labels = []
#         for d in data[2:]:
#             cycles_with_labels.append((d[0], d[1], d[2], file_id))
#         cycle_list.extend(cycles_with_labels)
#         file_id += 1
#         # break 
#     return cycle_list


if __name__ == "__main__":
    main()
