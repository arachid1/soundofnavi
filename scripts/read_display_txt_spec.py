import numpy as np
from matplotlib import pyplot as plt
import librosa
from librosa import display

def visualize_spectrogram(spect, sr, name):
    fig = plt.figure(figsize=(20, 10))
    display.specshow(
        spect,
        # y_axis="log",
        sr=sr,
        cmap="coolwarm"
    )
    plt.colorbar()
    plt.show()
    plt.savefig(name)

def main():
    version = 'v23'
    # perch_K07845-03.wav_F_319.txt, 1.0, 1.0
    # file_path = '../data/txt_datasets/all_sw_coch_preprocessed_v2_param_{}_augm_v0_cleaned_8000/perch_K07845-03.wav_F_319.txt'.format(version)
    # file_path = '../data/txt_datasets/all_sw_coch_preprocessed_v2_param_{}_augm_v0_cleaned_8000/perch_K07845-03.wav_F_1940.txt'.format(version)

    # file_path = '../data/txt_datasets/all_sw_coch_preprocessed_v2_param_{}_augm_v0_cleaned_8000/antwerp_RESPT_CALSA_ACT_exa_005_V1_POST_Thsr_31.txt'.format(version)
    file_path = '../data/txt_datasets/all_sw_coch_preprocessed_v2_param_{}_augm_v0_cleaned_8000/antwerp_RESPT_CALSA_ACT_exa_005_V1_POST_Thsr_100.txt'.format(version)
    # file_path = '../data/txt_datasets/all_sw_coch_preprocessed_v2_param_{}_augm_v0_cleaned_8000/antwerp_RESPT_CALSA_ACT_exa_005_V1_POST_Thsr_100.txt'.format(version)

    # antwerp_RESPT_CALSA_ACT_exa_005_V1_POST_Thsr_31

    spec = np.loadtxt(file_path, delimiter=',')
    visualize_spectrogram(spec, 8000, str("ant_2" + version))


if __name__ == "__main__":
    main()