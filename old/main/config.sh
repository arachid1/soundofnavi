#!/usr/bin/env bash
TESTING=0

export JOB_CAT=coch     #LOG OR MEL OR COCH
export MODEL=conv
export TRAIN_NUMBER=2412
export DESCRIPTION=audio_model

# DATASET
PREPROCESSED=v2
AUGM=v0
PARAM=v39

SIX=0
CONCAT=0

N_CLASSES=1
INITIAL_CHANNELS=1
EPSILON=1e-7

if [ "$CONCAT" = 1 ];
then
    INITIAL_CHANNELS=6
    SIX=1
fi

if [ "$JOB_CAT" = "mel" ]
then
    WIDTH=313
elif [ "$JOB_CAT" = "coch" ]
then
    WIDTH=1250
fi

HEIGHT=128

SR=8000
export FILE_NAME="all_sw_${JOB_CAT}_preprocessed_${PREPROCESSED}_param_${PARAM}_augm_${AUGM}_cleaned_$SR.pkl"

# normal params
N_EPOCHS=65
WEIGHT_DECAY=1e-3 # default: 1e-4 
LL2_REG=0
BATCH_SIZE=16
LR=1e-3 # default: 1e-3
MIN_LR=1e-6 # default: 1e-4
FACTOR=0.5 # default: 0.5
LR_PATIENCE=3 # defzult: 8 <- reffering to lr patience
ES_PATIENCE=20
MIN_DELTA=0
LABEL_SMOOTHING=0

CLAUSE=0
EPOCH_START=0
TARGET=1
CLASS_WEIGHTS=1
CUBEROOTING=1
NORMALIZING=1
OVERSAMPLE=0
ADAPTIVE_LR=1

JORDAN_DATASET=1
PNEUMONIA_ONLY=1

# augmentation params
AUGMENTATION=1
# {wav
WAV_ADD=0
WAV_Q=1000
WAV_PATH="v43"

# {spec}
SPEC_ADD=0
SPEC_Q=158
TIME_MASKING=0
FREQUENCY_MASKING=1
LOUDNESS=0

if [ "$TESTING" = 1 ];
then
    export DESCRIPTION=testing
fi

export PARAMS='{"N_CLASSES": '$N_CLASSES', "SR": '$SR', "BATCH_SIZE": '$BATCH_SIZE', "LR": '$LR', "SHAPE": ['$HEIGHT', '$WIDTH'], 
"LL2_REG": '$LL2_REG', "WEIGHT_DECAY": '$WEIGHT_DECAY', "N_EPOCHS": '$N_EPOCHS', "FACTOR": '$FACTOR', "LR_PATIENCE": '$LR_PATIENCE', "MIN_LR": '$MIN_LR', "LABEL_SMOOTHING": '$LABEL_SMOOTHING',
"EPSILON": '$EPSILON', "ES_PATIENCE": '$ES_PATIENCE', "OVERSAMPLE": '$OVERSAMPLE', "MIN_DELTA": '$MIN_DELTA', "PARAM": "'$PARAM'", "EPOCH_START": '$EPOCH_START', "JORDAN_DATASET": '$JORDAN_DATASET', "PNEUMONIA_ONLY": '$PNEUMONIA_ONLY', "CLAUSE": '$CLAUSE', "CUBEROOTING": '$CUBEROOTING', "NORMALIZING": '$NORMALIZING', "AUGMENTATION": '$AUGMENTATION', "TARGET": '$TARGET', "INITIAL_CHANNELS": '$INITIAL_CHANNELS', "ADAPTIVE_LR": '$ADAPTIVE_LR', "CLASS_WEIGHTS": '$CLASS_WEIGHTS', "SIX": '$SIX', "CONCAT": '$CONCAT', "TESTING": '$TESTING'}'
######################

export BUCKET_NAME=tf_learn_pattern_detection
export FOLDER=models

# export FILE=$(echo $FILE_NAME| cut -d'.' -f 1)
export FILE=${PARAM}
export DETAILS='__'$DESCRIPTION

export JOB_NAME=$MODEL'__'$TRAIN_NUMBER'__'$(date +%m_%H%M)'__'$FILE$DETAILS
export JOB_DIR=gs://$BUCKET_NAME/$JOB_CAT/$MODEL/$JOB_NAME
export TRAIN_FILE=gs://$BUCKET_NAME/datasets/$FILE_NAME

export MODULE_NAME=$MODEL.train$TRAIN_NUMBER
export PACKAGE_PATH=/Users/alirachidi/Documents/Sonavi_Labs/classification_algorithm/trainers/main/$FOLDER/$MODEL

export LOCAL_TRAIN_FILE=../../data/datasets/$FILE_NAME
export LOCAL_JOB_DIR=../../cache/pneumonia/$JOB_NAME

export REGION=us-central1

export SPEC_PARAMS='{"ADD": '$SPEC_ADD', "QUANTITY": '$SPEC_Q', "TIME_MASKING": '$TIME_MASKING', "FREQUENCY_MASKING": '$FREQUENCY_MASKING', "LOUDNESS": '$LOUDNESS'}'
export WAV_PARAMS='{"ADD": '$WAV_ADD', "QUANTITY": '$WAV_Q', "WAV_PATH": "'${WAV_PATH}'" }'

echo "File: " $MODULE_NAME
# echo "Category: " $JOB_CAT
echo "Description: " $DESCRIPTION
echo "Dataset: " $FILE_NAME
echo "Testing: " $TESTING
# echo "Number of Classes: " $N_CLASSES

read confirmation

printf "Six:  $SIX\n"
echo "Concatenate: " $CONCAT
echo "Clause: " $CLAUSE
# echo "Initial Channels: " $INITIAL_CHANNELS
printf "Use Class Weights: $CLASS_WEIGHTS\n"
echo "Cube Rooting: $CUBEROOTING "
echo "Normalizing: $NORMALIZING "
echo "Adaptive LR: $ADAPTIVE_LR "
echo "Target: $TARGET "
printf "Epoch Start: $EPOCH_START"

read confirmation

# printf "\nJordan Dataset: $JORDAN_DATASET "
# printf "\nPneumonia Only: $PNEUMONIA_ONLY "

# read confirmation


printf "\nNumber of epochs: $N_EPOCHS\n"
echo "Batch Size: $BATCH_SIZE"
echo "Weight Decay & LL2 REG: $WEIGHT_DECAY $LL2_REG "
echo "Label Smoothing: $LABEL_SMOOTHING "

read confirmation

printf "Learning Rate: $LR\n"
echo "Minimum Lr: $MIN_LR"
echo "Factor: $FACTOR "
echo "Plateau Patience: $PATIENCE "

printf "\nEarly Stopping Patience: $ES_PATIENCE\n"
echo "Min Delta: $MIN_DELTA"

read confirmation

printf "Augmentation: $AUGMENTATION\n"
printf "Audio augmentation params: $WAV_PARAMS\n"
printf "Spectrogram augmentation params: $SPEC_PARAMS\n"

read confirmation

