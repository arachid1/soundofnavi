#!/usr/bin/env bash

export JOB_CAT=coch     #LOG OR MEL OR COCH
export MODEL=conv
export TRAIN_NUMBER=2006
export DESCRIPTION=final_63_channels


SAVE=0
ADD_TUNED=0
OVERSAMPLE=0
CLASS_WEIGHTS=0

# DATASET
PREPROCESSED=v2
AUGM=v0
PARAM=v20

if [ "$JOB_CAT" = "mel" ]
then
    WIDTH=313
elif [ "$JOB_CAT" = "coch" ]
then
    WIDTH=1250
fi
HEIGHT=64
# WIDTH=250


SR=8000
export FILE_NAME="all_sw_${JOB_CAT}_preprocessed_${PREPROCESSED}_param_${PARAM}_augm_${AUGM}_cleaned_$SR.pkl"
# export FILE_NAME="perch_sw_${JOB_CAT}_param_${PARAM}_augm_${AUGM}_$SR.pkl"

DEFAULT=true

if [ "$DEFAULT" = true ];
then
    N_EPOCHS=35
    WEIGHT_DECAY=1e-4 # default: 1e-4 
    LL2_REG=0
    BATCH_SIZE=32
    LR=1e-3 # default: 1e-3
    MIN_LR=1e-5 # default: 1e-4
    FACTOR=0.5 # default: 0.5
    PATIENCE=5 # default: 8 <- reffering to lr patience
    ES_PATIENCE=7
    MIN_DELTA=0.01
else
    N_EPOCHS=35
    WEIGHT_DECAY=1e-4 # default: 1e-4 
    LL2_REG=0
    BATCH_SIZE=32
    LR=1e-3 # default: 1e-3
    MIN_LR=1e-5 # default: 1e-4
    FACTOR=0.5 # default: 0.5
    PATIENCE=1 # default: 8 <- reffering to lr patience
    ES_PATIENCE=7
    MIN_DELTA=0.01
fi


# HYPERPARAMETERS
N_CLASSES=2
INITIAL_CHANNELS=3
EPSILON=1e-7

export PARAMS='{"N_CLASSES": '$N_CLASSES', "SR": '$SR', "BATCH_SIZE": '$BATCH_SIZE', "LR": '$LR', "SHAPE": ['$HEIGHT', '$WIDTH'], 
"LL2_REG": '$LL2_REG', "WEIGHT_DECAY": '$WEIGHT_DECAY', "N_EPOCHS": '$N_EPOCHS', "FACTOR": '$FACTOR', "PATIENCE": '$PATIENCE', "MIN_LR": '$MIN_LR', 
"EPSILON": '$EPSILON', "ES_PATIENCE": '$ES_PATIENCE', "SAVE": '$SAVE', "ADD_TUNED": '$ADD_TUNED', "OVERSAMPLE": '$OVERSAMPLE', "MIN_DELTA": '$MIN_DELTA', "INITIAL_CHANNELS": '$INITIAL_CHANNELS', "CLASS_WEIGHTS": '$CLASS_WEIGHTS'}'
######################

export BUCKET_NAME=tf_learn_pattern_detection
export FOLDER=models

export FILE=$(echo $FILE_NAME| cut -d'.' -f 1)
export DETAILS='___'$DESCRIPTION'___'$TRAIN_NUMBER

export JOB_NAME=$MODEL'___'$(date +%m_%H%M)'___'$FILE$DETAILS
export JOB_DIR=gs://$BUCKET_NAME/$JOB_CAT/$MODEL/$JOB_NAME
export TRAIN_FILE=gs://$BUCKET_NAME/datasets/$FILE_NAME

export MODULE_NAME=$MODEL.train$TRAIN_NUMBER
export PACKAGE_PATH=/Users/alirachidi/Documents/Sonavi_Labs/classification_algorithm/trainers/main/$FOLDER/$MODEL

export LOCAL_TRAIN_FILE=../../data/datasets/$FILE_NAME
export LOCAL_JOB_DIR=../../cache/$JOB_NAME

export REGION=us-central1


echo "File: " $MODULE_NAME
echo "Category: " $JOB_CAT
echo "Description: " $DESCRIPTION
echo "Number of Classes: " $N_CLASSES
echo "Dataset: " $FILE_NAME

read confirmation

# echo "Learning Rate = $LR  -  Minimum Lr = $MIN_LR  -  Weight Decay = $WEIGHT_DECAY "
echo "Default: $DEFAULT"
echo "Number of epochs: $N_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Minimum Lr: $MIN_LR"
echo "Weight Decay & LL2 REG: $WEIGHT_DECAY $LL2_REG "

read confirmation

# echo "Save Images & Audios = $SAVE  -  Add Tuned Data = $ADD_TUNED  -  Oversample Minority Classes = $OVERSAMPLE"
echo "Save Images & Audios: $SAVE"
echo "Add Tuned Data: $ADD_TUNED"
echo "Oversample Minority Classes: $OVERSAMPLE"
echo "Use Class Weights: $CLASS_WEIGHTS "

read confirmation

