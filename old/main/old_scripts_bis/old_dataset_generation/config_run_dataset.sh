echo "-----------------------"

ALL=1
PERCH=0
ICBHI=0
ANTWERP=0

GCP=0
LOCAL=1

TEST=0
VISUALIZE=0

SR=8000
echo "Sample Rate: $SR"

##mel or log + parameters

COCH_PATH='../../cochlear_preprocessing/'
COCH_A='COCH_A.txt'
COCH_B='COCH_B.txt'
P='p.txt'

SPEC_TYPE="coch"

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

echo "Type of spectrogram: $SPEC_TYPE"
echo "number of fft : $N_FFT, hop length : $HOP_LENGTH, window length : $SPEC_WIN_LENGTH, number of mels: $N_MELS"
echo "Expected Dimensions: ($HEIGHT, $WIDTH)"

PROC_VERSION=v2 
PARAM_VERSION=v19
AUGM_VERSION=v0

# params
if [ "$PARAM_VERSION" = "v3" ]
then
    AUDIO_LENGTH=2
    STEP_SIZE=1
    OVERLAP_THRESHOLD=1
elif [ "$PARAM_VERSION" = "v4" ]
then
    AUDIO_LENGTH=2
    STEP_SIZE=1
    OVERLAP_THRESHOLD=0.5
else # v5, v6, v7, ...
    AUDIO_LENGTH=10
    STEP_SIZE=5
    OVERLAP_THRESHOLD=0.15
fi

# data access
FORMAT=sw # sliding window
ALL_FILE_DEST='../../data/datasets/all_'$FORMAT'_'$SPEC_TYPE'_preprocessed_'$PROC_VERSION'_param_'$PARAM_VERSION'_augm_'$AUGM_VERSION'_cleaned_'$SR'.pkl'

if [ "$ALL" = "1" ]
then
echo "Destination & Name: $ALL_FILE_DEST"
fi

ICBHI_FILE_DEST='../../data/datasets/icbhi_'$FORMAT'_'$SPEC_TYPE'_preprocessed_'$PROC_VERSION'_param_'$PARAM_VERSION'_augm_'$AUGM_VERSION'_'$SR'.pkl'
ICBHI_ROOT='../../data/Data-CVSDEncoded/ICBHI/'
# ICBHI_ROOT='../../data/m4a_raw_audios/icbhi_preprocessed_'$PROC_VERSION'_cleaned_'$SR'/'
# ICBHI_ROOT='../../data/raw_audios/icbhi_preprocessed_'$PROC_VERSION'_cleaned_'$SR'/'

PERCH_FILE_DEST='../../data/datasets/perch_'$FORMAT'_'$SPEC_TYPE'_param_'$PARAM_VERSION'_augm_'$AUGM_VERSION'_'$SR'.pkl'
PERCH_ROOT='../../data/Data-CVSDEncoded/PERCH/'
# PERCH_ROOT='../../data/m4a_raw_audios/perch_'$SR'_10seconds/'
# PERCH_ROOT='../../data/raw_audios/perch_'$SR'_10seconds/'

ANTWERP_FILE_DEST='../../data/datasets/antwerp_'$FORMAT'_'$SPEC_TYPE'_param_'$PARAM_VERSION'_augm_'$AUGM_VERSION'_'$SR'.pkl'
ANTWERP_ROOT='../../data/Data-CVSDEncoded/AntwerpClinical/'
# ANTWERP_ROOT='../../data/m4a_raw_audios/Antwerp_Clinical/'
# ANTWERP_ROOT='../../data/raw_audios/Antwerp_Clinical/'

ANTWERP_SIMULATED_ROOT='../../data/Data-CVSDEncoded/AntwerpSimulated/'
# ANTWERP_SIMULATED_ROOT='../../data/m4a_raw_audios/Antwerp_Simulated/'
# ANTWERP_SIMULATED_ROOT='../../data/raw_audios/Antwerp_Simulated/'

AUGMENTATION=0

WAV_ADD=0
WAV_RATIO=0.2
GAUSSIAN_NOISE=0
VTLP=0
SHIFT=0
MIN_SNR=0.001
MAX_SNR=1
WAV_PARAMS='{"add": '$WAV_ADD', "ratio": '$WAV_RATIO', "gaussian_noise": '$GAUSSIAN_NOISE', "vtlp": '$VTLP', "shift": '$SHIFT', "min_snr": '$MIN_SNR', "max_snr": '$MAX_SNR'}'

SPEC_ADD=0
SPEC_RATIO=0.3
TIME_MASKING=0
FREQUENCY_MASKING=0

SPEC_PARAMS='{"add": '$SPEC_ADD', "ratio": '$SPEC_RATIO', "time_masking": '$TIME_MASKING', "frequency_masking": '$FREQUENCY_MASKING'}'

export DATASET_PARAMS='{"SR": '$SR', "SPEC_TYPE": "'$SPEC_TYPE'", "GCP": '$GCP', "LOCAL": '$LOCAL', "ALL": '$ALL', "PERCH": '$PERCH', "ICBHI": '$ICBHI', "ANTWERP": '$ANTWERP', "COCH_A": "'$COCH_A'", "COCH_B": "'$COCH_B'", "P": "'$P'", "AUGMENTATION": '$AUGMENTATION',
"HEIGHT": '$HEIGHT', "WIDTH": '$WIDTH', "ANTWERP_SIMULATED_ROOT": "'$ANTWERP_SIMULATED_ROOT'",  "N_FFT": '$N_FFT', "HOP_LENGTH": '$HOP_LENGTH', "SPEC_WIN_LENGTH": '$SPEC_WIN_LENGTH', "N_MELS": '$N_MELS', "AUDIO_LENGTH": '$AUDIO_LENGTH', 
"STEP_SIZE": '$STEP_SIZE', "OVERLAP_THRESHOLD": '$OVERLAP_THRESHOLD', "TEST": '$TEST', "BP": '$BP', "FS": '$FS', "COCH_PATH": "'$COCH_PATH'", "ICBHI_ROOT": "'$ICBHI_ROOT'", "PERCH_ROOT": "'$PERCH_ROOT'", 
"ANTWERP_ROOT": "'$ANTWERP_ROOT'", "ALL_FILE_DEST": "'$ALL_FILE_DEST'", "ICBHI_FILE_DEST" : "'$ICBHI_FILE_DEST'", "PERCH_FILE_DEST": "'$PERCH_FILE_DEST'", "ANTWERP_FILE_DEST": "'$ANTWERP_FILE_DEST'"}'

echo "PARAMS: $DATASET_PARAMS"
echo "-----------------------"

python -W ignore -u generate_dataset.py --params "$DATASET_PARAMS" --wav_params "$WAV_PARAMS" --spec_params "$SPEC_PARAMS"

echo "Done."
