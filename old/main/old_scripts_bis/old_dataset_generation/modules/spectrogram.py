from scipy.signal import lfilter
import librosa
from librosa import display, resample
import numpy as np


def generate_spectrograms(cycles, spec_type, sr, n_fft, hop_length, spec_win_length, n_mels, coch_path, coch_params, train_test_ratio=0.2):

    print("Generating spectograms...")
    # convert
    for i in range(0, len(cycles)):
        for y in range(0, len(cycles[i])):
            cycles[i][y] = convert_to_spec(
                cycles[i][y], spec_type, sr, n_fft, hop_length, spec_win_length, n_mels, coch_path, coch_params)
            np.random.shuffle(cycles[i][y])

    return cycles


def generate_cochlear_spec(x, coch_params, coch_b, coch_a, order):

    # get filter bank
    L, M = coch_b.shape
    # print("Shape: [{}, {}]".format(L, M))
    # exit()
    L_x = len(x)

    # octave shift, nonlinear factor, frame length, leaky integration
    shft = coch_params["shft"]  # octave shift (Matlab index: 4)
    fac = coch_params["fac"]  # nonlinear factor (Matlab index: 3)
    frmlen = coch_params["frmlen"]  # (Matlab index: 1)
    L_frm = np.round(frmlen * (2**(4+shft)))  # frame length (points)
    tc = coch_params["tc"]  # time constant (Matlab index: 2)
    tc = np.float64(tc)
    alph_exponent = -(1/(tc*(2**(4+shft))))
    alph = np.exp(alph_exponent)  # decaying factor

    # get data, allocate memory for output
    N = np.ceil(L_x/L_frm)  # number of frames
    # print("Number of frames: {}".format(N))
    # x = generate_padded_samples(x, ) # TODO: come back to padding

    v5 = np.zeros([int(N), M - 1])

    ##############################
    # last channel (highest frequency)
    ##############################

    # get filters from stored matrix
    # print("Number of filters: {}".format(M))
    p = int(order[M-1])  # ian's change 11/6

    # ian changed to 0 because of the way p, coch_a, and coch_b are seperated
    B = coch_b[0:p+1, M - 1]  # M-1 before
    A = coch_a[0:p+1, M - 1]
    y1 = lfilter(B, A, x)
    y2 = y1
    y2_h = y2

    # All other channels
    for ch in range(M-2, -1, -1):

        # ANALYSIS: cochlear filterbank
        # IIR: filter bank convolution ---> y1
        p = int(order[ch])
        B = coch_b[0:p+1, ch]
        A = coch_a[0:p+1, ch]

        y1 = lfilter(B, A, x)

        # TRANSDUCTION: hair cells
        y2 = y1

        # REDUCTION: lateral inhibitory network
        # masked by higher (frequency) spatial response
        y3 = y2 - y2_h
        y2_h = y2

        # half-wave rectifier
        y4 = y3.copy()
        y4[y4 < 0] = 0

        # temporal integration window #
        # leaky integration. alternative could be simpler short term average
        y5 = lfilter([1], [1-alph], y4)

        v5_row = []
        for i in range(1, int(N) + 1):
            v5_row.append(y5[int(L_frm*i) - 1])
        v5[:, ch] = v5_row
        # v5[:, ch] = y5[int(L_frm):int(L_frm*N)] # N elements, each space with L_frm
    # print("...End")

    return v5


def convert_to_log(data, n_fft, hop_length, spec_win_length):
    spec_list = []
    for d in data:
        log_spectrogram = librosa.power_to_db(
            np.abs(librosa.stft(d[0], n_fft=n_fft, hop_length=hop_length, win_length=spec_win_length,
                                center=True, window='hann')) ** 2, ref=1.0)
        spec_list.append((log_spectrogram, d[1], d[2]))
    return spec_list


def convert_to_coch(data, coch_path, coch_params):
    spec_list = []
    for d in data:
        if (not (len(d[0]) == 80000)):
            continue
        coch_a = np.loadtxt(coch_path + coch_params["COCH_A"], delimiter=',')
        coch_b = np.loadtxt(coch_path + coch_params["COCH_B"], delimiter=',')
        order = np.loadtxt(coch_path + coch_params["P"], delimiter=',')
        coch_spectrogram = generate_cochlear_spec(
            d[0], coch_params, coch_b, coch_a, order)
        coch_spectrogram = np.transpose(coch_spectrogram)
        spec_list.append((coch_spectrogram, d[1], d[2]))
        if len(spec_list) % 2000 == 0:
            print("Completed {} elements...".format(len(spec_list)))
    return spec_list


def convert_to_mel(data, sr, n_fft, hop_length, n_mels):
    spec_list = []
    for d in data:
        mel_spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(
            d[0], sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels), ref=np.max)
        spec_list.append((mel_spectrogram, d[1], d[2]))
    return spec_list


def convert_to_spec(data, spec_type, sr, n_fft, hop_length, spec_win_length, n_mels, coch_path, coch_params):
    n_fft = n_fft
    hop_length = hop_length
    if spec_type == "log":
        spec_list = convert_to_log(data, n_fft, hop_length, spec_win_length)
    elif spec_type == "coch":
        spec_list = convert_to_coch(data, coch_path, coch_params)
    else:
        spec_list = convert_to_mel(data, sr, n_fft, hop_length, n_mels)
    return spec_list
