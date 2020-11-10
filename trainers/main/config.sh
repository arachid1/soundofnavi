#!/usr/bin/env bash

export JOB_CAT=log #LOG OR MEL
SR=8000

#JOB

export MODEL=conv #LSTM OR CONV
export TRAIN_NUMBER=305
export DESCRIPTION=diffparams_save_oversample

# JOB PARAMS

SAVE=1
ADD_TUNED=0
OVERSAMPLE=1
CLASS_WEIGHTS=1

[ "$JOB_CAT" = log ] && HEIGHT=257 || HEIGHT=128 
[ "$SR" = 8000 ] && WIDTH=251 || WIDTH=126
# WIDTH=63

# HYPERPARAMETERS
N_CLASSES=2
INITIAL_CHANNELS=3
EPSILON=1e-7

DEFAULT=true

if [ "$DEFAULT" = true ];
then
    N_EPOCHS=55
    WEIGHT_DECAY=1e-4 # default: 1e-4
    LL2_REG=0
    BATCH_SIZE=64
    LR=1e-3 # default: 1e-3
    MIN_LR=1e-4 # default: 1e-4
    FACTOR=0.5 # default: 0.5
    PATIENCE=8 # default: 8 <- reffering to lr patience
    ES_PATIENCE=15
    MIN_DELTA=0.01
else
    N_EPOCHS=55
    WEIGHT_DECAY=1e-5 # default: 1e-4
    LL2_REG=0
    BATCH_SIZE=64
    LR=5e-4 # default: 1e-3
    MIN_LR=1e-4 # default: 1e-4
    FACTOR=0.75 # default: 0.5
    PATIENCE=8 # default: 8 <- reffering to lr patience
    ES_PATIENCE=16
    MIN_DELTA=0.01
fi

# DATASET

PREPROCESSED=v2
PARAM=v3
AUGM=v0

# "perch_sw_${JOB_CAT}_param_${PARAM}_augm_${AUGM}_$SR.pkl"
export FILE_NAME="all_sw_${JOB_CAT}_preprocessed_${PREPROCESSED}_param_${PARAM}_augm_${AUGM}_$SR.pkl"

export PARAMS='{"N_CLASSES": '$N_CLASSES', "SR": '$SR', "BATCH_SIZE": '$BATCH_SIZE', "LR": '$LR', "SHAPE": ['$HEIGHT', '$WIDTH'], 
"LL2_REG": '$LL2_REG', "WEIGHT_DECAY": '$WEIGHT_DECAY', "N_EPOCHS": '$N_EPOCHS', "FACTOR": '$FACTOR', "PATIENCE":'$PATIENCE', "MIN_LR":'$MIN_LR', 
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

export LOCAL_TRAIN_FILE=../../../data/datasets/$FILE_NAME
export LOCAL_JOB_DIR=../../../cache/datasets/$JOB_NAME

export REGION=us-central1


echo "File: " $MODULE_NAME
echo "Category: " $JOB_CAT
echo "Description: " $DESCRIPTION
echo "Number of Classes: " $N_CLASSES
echo "Dataset: " $FILE_NAME


read confirmation

# echo "Learning Rate = $LR  -  Minimum Lr = $MIN_LR  -  Weight Decay = $WEIGHT_DECAY "
echo "Number of Epochs: " $N_EPOCHS
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Minimum Lr: $MIN_LR"
echo "Weight Decay & LL2 REG: $WEIGHT_DECAY $LLE_REG "

read confirmation

# echo "Save Images & Audios = $SAVE  -  Add Tuned Data = $ADD_TUNED  -  Oversample Minority Classes = $OVERSAMPLE"
echo "Save Images & Audios: $SAVE"
echo "Add Tuned Data: $ADD_TUNED"
echo "Oversample Minority Classes: $OVERSAMPLE"
echo "Use Class Weights: $CLASS_WEIGHTS "

read confirmation

MASTER_TYPE=n1-standard-32
MACHINE=NVIDIA_TESLA_P4
COUNT=2
echo "Master Type: $MASTER_TYPE "
echo "Machine : $MACHINE "
echo "Machine Count: $COUNT "

read confirmation

# MASTER_TYPE=n1-standard-32
# MACHINE=NVIDIA_TESLA_P4
# COUNT=2

# export YAML_CONFIG='{
#     "trainingInput": {
#       "scaleTier": "CUSTOM",
#       "masterType": "'$MASTER_TYPE'",
#       "masterConfig": {
#           "acceleratorConfig": {
#               "count": "'$COUNT'",
#               "type": "'$MACHINE'"
#           }
#       },
#       "runtimeVersion": "2.1",
#       "pythonVersion": "3.7",
#     }
#   }'
