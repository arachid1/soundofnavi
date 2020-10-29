echo "-----------------------"

ALL=1
SAVE_IND=0 # if All, then should i save ind?
SAVE_ANY=0

PERCH=0
ICBHI=0
TEST=0

SR=8000
echo "Sample Rate: $SR"

##mel or log + parameters
SPEC_TYPE="log"
if [ "$SPEC_TYPE" = "log" ]
then
    N_FFT=512
    HOP_LENGTH=64 
    SPEC_WIN_LENGTH=160
    N_MELS=-1
    HEIGHT=257
    if [ "$SR" = "8000" ]
    then
        WIDTH=251
    else
        WIDTH=126
    fi
else #mel
    N_FFT=1024
    HOP_LENGTH=256
    N_MELS=128
    SPEC_SPEC_WIN_LENGTH=-1
    if [ "$SR" = "8000" ]
    then
        HEIGHT=128
        WIDTH=63
    fi
fi

echo "Type of spectrogram: $SPEC_TYPE"
echo "number of fft : $N_FFT, hop length : $HOP_LENGTH, window length : $SPEC_WIN_LENGTH, number of mels: $N_MELS"


echo "Expected Dimensions: ($HEIGHT, $WIDTH)"

# perch, icbhi or all
if [ "$ALL" = "1" ]
then
    echo "Do you want to compile all the datasets: Yes"
    COMB=all
    NUM_ELEMENTS=24589
else
    echo "Do you want to compile all the datasets: No"
    if [ "$PERCH" = "1" ]
    then
        echo "Only PERCH: True"
        echo "Only ICBHI: False"
        COMB=perch
        NUM_ELEMENTS=6678
    else
        echo "Only PERCH: False"
        echo "Only ICBHI: True"
        COMB=icbhi
        NUM_ELEMENTS=17911

    fi
fi

echo "Number of Elements: $NUM_ELEMENTS"

PROC_VERSION=v2 
PARAM_VERSION=v2
# param v2 -> generating specs with different values for n_fft, hop_length, etc 
# param v3 -> using 1 seconds instead of 0.5 second as overlap + clean ICBHI
AUGM_VERSION=v0

FORMAT=sw

# data access
ALL_FILE_DEST='/Users/alirachidi/Documents/Sonavi_Labs/classification_algorithm/data/datasets/all_'$FORMAT'_'$SPEC_TYPE'_preprocessed_'$PROC_VERSION'_param_'$PARAM_VERSION'_augm_'$AUGM_VERSION'_cleaned_'$SR'.pkl'

if [ "$ALL" = "1" ]
then
echo "Destination & Name: $ALL_FILE_DEST"
fi

ICBHI_FILE_DEST='/Users/alirachidi/Documents/Sonavi_Labs/classification_algorithm/data/datasets/icbhi_'$FORMAT'_'$SPEC_TYPE'_preprocessed_'$PROC_VERSION'_param_'$PARAM_VERSION'_augm_'$AUGM_VERSION'_'$SR'.pkl'
# ICBHI_ROOT='../../data/raw_audios/icbhi_preprocessed_'$PROC_VERSION'_'$SR'/'
ICBHI_ROOT='/Users/alirachidi/Documents/Sonavi_Labs/classification_algorithm/data/raw_audios/icbhi_preprocessed_'$PROC_VERSION'_cleaned_'$SR'/'

PERCH_FILE_DEST='/Users/alirachidi/Documents/Sonavi_Labs/classification_algorithm/data/datasets/perch_'$FORMAT'_'$SPEC_TYPE'_param_'$PARAM_VERSION'_augm_'$AUGM_VERSION'_'$SR'.pkl'
PERCH_ROOT='/Users/alirachidi/Documents/Sonavi_Labs/classification_algorithm/data/raw_audios/perch_'$SR'/'


# params
AUDIO_LENGTH=2
STEP_SIZE=1
OVERLAP_THRESHOLD=0.5

# augmentation
AUGMENTATION=0
TIME_STRETCHING=0
TIME_WARPING=0
PITCH_SHIFTING=0

# TARGET=

export DATASET_PARAMS='{"SR": '$SR', "SPEC_TYPE": "'$SPEC_TYPE'", "ALL": '$ALL', "SAVE_IND": '$SAVE_IND', "SAVE_ANY": '$SAVE_ANY', "PERCH": '$PERCH', "ICBHI": '$ICBHI', "HEIGHT": '$HEIGHT', "WIDTH": '$WIDTH', "N_FFT": '$N_FFT', "HOP_LENGTH": '$HOP_LENGTH', "SPEC_WIN_LENGTH": '$SPEC_WIN_LENGTH', "N_MELS": '$N_MELS', "AUDIO_LENGTH": '$AUDIO_LENGTH', "STEP_SIZE": '$STEP_SIZE', "OVERLAP_THRESHOLD": '$OVERLAP_THRESHOLD', "AUGMENTATION": '$AUGMENTATION', "TIME_STRETCHING": '$TIME_STRETCHING', "PITCH_SHIFTING": '$PITCH_SHIFTING', "TIME_WARPING": '$TIME_WARPING', "TEST": '$TEST', "ICBHI_ROOT": "'$ICBHI_ROOT'", "PERCH_ROOT": "'$PERCH_ROOT'", "ALL_FILE_DEST": "'$ALL_FILE_DEST'", "ICBHI_FILE_DEST" : "'$ICBHI_FILE_DEST'","PERCH_FILE_DEST": "'$PERCH_FILE_DEST'"}'

echo "PARAMS: $DATASET_PARAMS"
echo "-----------------------"

sleep 10

/opt/anaconda3/envs/ObjectDetection/bin/python generate_dataset.py --params "$DATASET_PARAMS"