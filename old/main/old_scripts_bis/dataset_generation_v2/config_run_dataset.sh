 echo "-----------------------"


PROC_VERSION=v2 
export PARAM_VERSION=v49
AUGM_VERSION=v0

SR=8000
echo "Sample Rate: $SR"

COCH_PATH='../../cochlear_preprocessing/128_channels/'
COCH_A='COCH_A.txt'
COCH_B='COCH_B.txt'
P='p.txt'

SPEC_TYPE="mel"

if [ "$SPEC_TYPE" = "log" ]
then
    N_FFT=512
    HOP_LENGTH=64 
    SPEC_WIN_LENGTH=160
    N_MELS=-1
    HEIGHT=257
    WIDTH=251
elif [ "$SPEC_TYPE" = "mel" ]
then
    N_FFT=1024
    HOP_LENGTH=256
    N_MELS=128
    SPEC_WIN_LENGTH=-1
    HEIGHT=128
    WIDTH=63
elif [ "$SPEC_TYPE" = "linear" ]
then
    N_FFT=512
    HOP_LENGTH=64
    N_MELS=-1
    SPEC_WIN_LENGTH=-1
    HEIGHT=128 #CHANGE
    WIDTH=63
elif [ "$SPEC_TYPE" = "coch" ]
then
    N_FFT=-1
    HOP_LENGTH=-1
    N_MELS=-1
    SPEC_WIN_LENGTH=-1
    HEIGHT=128 #CHANGE
    WIDTH=1250
fi

BP=1
FS=8000
TEST=0
VISUALIZE=0

echo "Type of spectrogram: $SPEC_TYPE"
echo "number of fft : $N_FFT, hop length : $HOP_LENGTH, window length : $SPEC_WIN_LENGTH, number of mels: $N_MELS"
echo "Expected Dimensions: ($HEIGHT, $WIDTH)"

AUDIO_LENGTH=10
STEP_SIZE=5
OVERLAP_THRESHOLD=0.15 # ratio
LENGTH_THRESHOLD=0.5 # ratio 

# data access
FORMAT=sw # sliding window
ALL_FILE_DEST='../../data/datasets/all_'$FORMAT'_'$SPEC_TYPE'_preprocessed_'$PROC_VERSION'_param_'$PARAM_VERSION'_augm_'$AUGM_VERSION'_cleaned_'$SR'.pkl'

echo "Destination & Name: $ALL_FILE_DEST"

# ICBHI_ROOT='../../data/Data-CVSDEncoded_highercutoff/ICBHI/'
# ICBHI_ROOT='../../data/Data-CVSDEncoded/ICBHI/'
# ICBHI_ROOT='../../data/m4a_raw_audios/icbhi_preprocessed_'$PROC_VERSION'_cleaned_'$SR'/'
ICBHI_ROOT='../../data/raw_audios/icbhi_preprocessed_'$PROC_VERSION'_cleaned_'$SR'/'

# PERCH_ROOT='../../data/Data-CVSDEncoded_highercutoff/PERCH/'
# PERCH_ROOT='../../data/Data-CVSDEncoded/PERCH/'
# PERCH_ROOT='../../data/m4a_raw_audios/perch_'$SR'_10seconds/'
PERCH_ROOT='../../data/raw_audios/perch_'$SR'_10seconds/'

# ANTWERP_ROOT='../../data/Data-CVSDEncoded_highercutoff/AntwerpClinical/'
# ANTWERP_ROOT='../../data/Data-CVSDEncoded/AntwerpClinical/'
# ANTWERP_ROOT='../../data/m4a_raw_audios/Antwerp_Clinical/'
ANTWERP_ROOT='../../data/raw_audios/Antwerp_Clinical_Complete/'

# ANTWERP_SIMULATED_ROOT='../../data/Data-CVSDEncoded_highercutoff/AntwerpSimulated/'
# ANTWERP_SIMULATED_ROOT='../../data/Data-CVSDEncoded/AntwerpSimulated/'
# ANTWERP_SIMULATED_ROOT='../../data/m4a_raw_audios/Antwerp_Simulated/'
ANTWERP_SIMULATED_ROOT='../../data/raw_audios/Antwerp_Simulated/'

BANGLADESH_ROOT='../../data/PCV_SEGMENTED_Processed_Files/'

AUGMENTATION=0
# GAUSSIAN_NOISE=0
# MIN_SNR=0.001
# MAX_SNR=0
DRC=0
# SHIFT=0
TIME_STRETCHING=1
PITCH_SHIFTING=1
# add pitch shifting factor
# add time stretch factor
WAV_PARAMS='{"PITCH_SHIFTING": '$PITCH_SHIFTING', "TIME_STRETCHING": '$TIME_STRETCHING', "DRC": '$DRC'}'

SPEC_ADD=0
SPEC_Q=0
TIME_MASKING=0
FREQUENCY_MASKING=0
SPEC_PARAMS='{"ADD": '$SPEC_ADD', "QUANTITY": '$SPEC_Q', "TIME_MASKING": '$TIME_MASKING', "FREQUENCY_MASKING": '$FREQUENCY_MASKING'}'

export DATASET_PARAMS='{"SR": '$SR', "SPEC_TYPE": "'$SPEC_TYPE'", "COCH_A": "'$COCH_A'", "COCH_B": "'$COCH_B'", "P": "'$P'", "AUGMENTATION": '$AUGMENTATION',
"HEIGHT": '$HEIGHT', "WIDTH": '$WIDTH',  "N_FFT": '$N_FFT', "HOP_LENGTH": '$HOP_LENGTH', "SPEC_WIN_LENGTH": '$SPEC_WIN_LENGTH', "N_MELS": '$N_MELS', 
"AUDIO_LENGTH": '$AUDIO_LENGTH', "STEP_SIZE": '$STEP_SIZE', "OVERLAP_THRESHOLD": '$OVERLAP_THRESHOLD', "LENGTH_THRESHOLD": '$LENGTH_THRESHOLD', "TEST": '$TEST', "BP": '$BP', "FS": '$FS', "COCH_PATH": "'$COCH_PATH'", 
"ANTWERP_SIMULATED_ROOT": "'$ANTWERP_SIMULATED_ROOT'", "ICBHI_ROOT": "'$ICBHI_ROOT'", "PERCH_ROOT": "'$PERCH_ROOT'", "ANTWERP_ROOT": "'$ANTWERP_ROOT'", "BANGLADESH_ROOT": "'$BANGLADESH_ROOT'",
"ALL_FILE_DEST": "'$ALL_FILE_DEST'"}'

printf "\nGeneral Parameters:\n $DATASET_PARAMS \n"
printf "\nAudio Augm. Parameters:\n $WAV_PARAMS \n"
printf "\nSpectrogram Augm. Parameters:\n $SPEC_PARAMS \n"
echo "-----------------------"

# python -W ignore -u generate_dataset.py --params "$DATASET_PARAMS" --wav_params "$WAV_PARAMS" --spec_params "$SPEC_PARAMS"
# python -W ignore -u generate_bangladesh.py --params "$DATASET_PARAMS" --wav_params "$WAV_PARAMS" --spec_params "$SPEC_PARAMS"
python -W ignore -u generate_pneumonia_icbhi.py --params "$DATASET_PARAMS" --wav_params "$WAV_PARAMS" --spec_params "$SPEC_PARAMS"
# python -W ignore -u generate_jordan.py --params "$DATASET_PARAMS" --wav_params "$WAV_PARAMS" --spec_params "$SPEC_PARAMS"

echo "Done."
