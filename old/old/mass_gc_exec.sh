#!/usr/bin/env bash

######################
#  parameters that are often changed

SOURCE=222
EXPERIMENT_NB=1
EXPERIMENT='exp_'$SOURCE'_'$EXPERIMENT_NB''
LOCAL=true 

##### log or mel

MASS_JOB_CAT=log
MASS_FOLDER=models
MASS_MODEL=conv

#####

MASS_MODULE_NAME=$MASS_MODEL.train$SOURCE
#perch_sw_log_param_v2_8000.pkl
SR=8000
# #icbhi_sw_log_spec_v1_4000 or domain_sw_log_8000
MASS_FILE_NAME="all_sw_log_preprocessed_v2_param_v2_$SR.pkl"
WIDTH=251
N_EPOCHS=100
N_CLASSES=2
EPSILON=1e-7
ES_PATIENCE=10
MIN_DELTA=0.01
INITIAL_CHANNELS=3

SAVE=0
ADD_TUNED=0
OVERSAMPLE=0
CLASS_WEIGHTS=0
# Set ll2 or weight decay to 0

printf "\nCategory: " $MASS_JOB_CAT
echo "Model: " $MASS_MODEL
echo "Dataset: " $MASS_FILE_NAME
echo "Width: " $WIDTH
echo "Folder: " $MASS_FOLDER
echo "Experiment: " $EXPERIMENT
echo "Module: " $MASS_MODULE_NAME

# read confirmation

######################
# parameters that are dynamic (i.e., descriptions)
declare -A PARAMS_DESCS
# n_classes, sr, batch_size, lr, shape, ll2_reg, weight_decay, n_epochs, factor, patience, min_lr, epsilon, es_patience, save, add_tuned, oversample, min_delata, initial_channels, class_weights
PARAMS_DESCS[0]="$N_CLASSES $SR 64 0.001 [257, $WIDTH] 0 1e-3 $N_EPOCHS 0.5 1e-4 $EPSILON $ES_PATIENCE $SAVE $ADD_TUNED $OVERSAMPLE $MIN_DELTA $INITIAL_CHANNELS $CLASS_WEIGHTS"
PARAMS_DESCS[1]="base"

PARAMS_DESCS[2]="$N_CLASSES $SR 64 0.001 [257, $WIDTH] 0 1e-3 $N_EPOCHS 0.25 1e-4 $EPSILON $ES_PATIENCE $SAVE $ADD_TUNED $OVERSAMPLE $MIN_DELTA $INITIAL_CHANNELS $CLASS_WEIGHTS"
PARAMS_DESCS[3]="base_factor_0.5"


# declare -A PARAMS_LIST=()

for ((i = 0; i < ${#PARAMS_DESCS[@]}; ++i));
do
    PARAMS=${PARAMS_DESCS[$i]}
    DESC=${PARAMS_DESCS[$((i+1))]}
    z=${PARAMS[2]}
    i=$((i+1))

done



#     echo "Name of Job $i: ${MASS_JOB_NAMES[$i]}" 
#     echo "Params of Job $i: ${PARAMS_LIST[$i]}" 

# echo ${#PARAMS_DESCS[@]}

# echo ${NUMBERED_PARAMS_LIST[P]}
# echo ${NUMBERED_PARAMS_LIST[D]}

# for ((i = 0; i < ${#NUMBERED_PARAMS_LIST[P][@]}; ++i));
#     echo "${NUMBERED_PARAMS_LIST[$i]}"
# echo $PARAMS_LIST

# append arrays containing number values and description individually

# for each list: 
# - pass the indexed elements of the list to a "params" string
#
# - passes the paraneters [dict of arrays?] by index to a string that is pre-placed to take the parameters
# - add the params and description to the list 


# declare -A PARAMS_LIST=()
# declare -A DESCRIPTIONS=()

# PARAMS='{"N_CLASSES": '$N_CLASSES', "SR": '$SR', "BATCH_SIZE": 64, "LR": 0.001, "SHAPE": [257, '$WIDTH'], 
# "LL2_REG": 0.000, "WEIGHT_DECAY":0.0001, "N_EPOCHS": '$N_EPOCHS', "FACTOR": 0.5, "PATIENCE":5, "MIN_LR":0.0001, "EPSILON": '$EPSILON', "ES_PATIENCE": '$ES_PATIENCE', "SAVE": '$SAVE', "ADD_TUNED": '$ADD_TUNED', "OVERSAMPLE": '$OVERSAMPLE', "MIN_DELTA": '$MIN_DELTA', "INITIAL_CHANNELS": '$INITIAL_CHANNELS', "CLASS_WEIGHTS": '$CLASS_WEIGHTS'}'
# DESC='regular'

# PARAMS_LIST[${#PARAMS_LIST[@]}]=$PARAMS
# DESCRIPTIONS[${#DESCRIPTIONS[@]}]=$DESC

# PARAMS='{"N_CLASSES": '$N_CLASSES', "SR": '$SR', "BATCH_SIZE": 64, "LR": 0.001, "SHAPE": [257, '$WIDTH'], 
# "LL2_REG": 0.000, "WEIGHT_DECAY": 0.0001, "N_EPOCHS": '$N_EPOCHS', "FACTOR": 0.25, "PATIENCE":5, "MIN_LR":0.0001, "EPSILON": '$EPSILON', "ES_PATIENCE": '$ES_PATIENCE', "SAVE": '$SAVE', "ADD_TUNED": '$ADD_TUNED', "OVERSAMPLE": '$OVERSAMPLE', "MIN_DELTA": '$MIN_DELTA', "INITIAL_CHANNELS": '$INITIAL_CHANNELS', "CLASS_WEIGHTS": '$CLASS_WEIGHTS'}'
# DESC='factor025'

# PARAMS_LIST[${#PARAMS_LIST[@]}]=$PARAMS
# DESCRIPTIONS[${#DESCRIPTIONS[@]}]=$DESC

# PARAMS='{"N_CLASSES": '$N_CLASSES', "SR": '$SR', "BATCH_SIZE": 64, "LR": 0.001, "SHAPE": [257, '$WIDTH'], 
# "LL2_REG": 0.000, "WEIGHT_DECAY": 0.0001, "N_EPOCHS": '$N_EPOCHS', "FACTOR": 0.75, "PATIENCE":5, "MIN_LR":0.0001, "EPSILON": '$EPSILON', "ES_PATIENCE": '$ES_PATIENCE', "SAVE": '$SAVE', "ADD_TUNED": '$ADD_TUNED', "OVERSAMPLE": '$OVERSAMPLE', "MIN_DELTA": '$MIN_DELTA', "INITIAL_CHANNELS": '$INITIAL_CHANNELS', "CLASS_WEIGHTS": '$CLASS_WEIGHTS'}'
# DESC='factor075'

# PARAMS_LIST[${#PARAMS_LIST[@]}]=$PARAMS
# DESCRIPTIONS[${#DESCRIPTIONS[@]}]=$DESC

# PARAMS='{"N_CLASSES": '$N_CLASSES', "SR": '$SR', "BATCH_SIZE": 128, "LR": 0.001, "SHAPE": [257, '$WIDTH'], 
# "LL2_REG": 0.000, "WEIGHT_DECAY": 0.0001, "N_EPOCHS": '$N_EPOCHS', "FACTOR": 0.5, "PATIENCE":5, "MIN_LR":0.0001, "EPSILON": '$EPSILON', "ES_PATIENCE": '$ES_PATIENCE', "SAVE": '$SAVE', "ADD_TUNED": '$ADD_TUNED', "OVERSAMPLE": '$OVERSAMPLE', "MIN_DELTA": '$MIN_DELTA', "INITIAL_CHANNELS": '$INITIAL_CHANNELS', "CLASS_WEIGHTS": '$CLASS_WEIGHTS'}'
# DESC='regular_bsize128'

# PARAMS_LIST[${#PARAMS_LIST[@]}]=$PARAMS
# DESCRIPTIONS[${#DESCRIPTIONS[@]}]=$DESC

# PARAMS='{"N_CLASSES": '$N_CLASSES', "SR": '$SR', "BATCH_SIZE": 128, "LR": 0.001, "SHAPE": [257, '$WIDTH'], 
# "LL2_REG": 0.000, "WEIGHT_DECAY": 0.0001, "N_EPOCHS": '$N_EPOCHS', "FACTOR": 0.25, "PATIENCE":5, "MIN_LR":0.0001, "EPSILON": '$EPSILON', "ES_PATIENCE": '$ES_PATIENCE', "SAVE": '$SAVE', "ADD_TUNED": '$ADD_TUNED', "OVERSAMPLE": '$OVERSAMPLE', "MIN_DELTA": '$MIN_DELTA', "INITIAL_CHANNELS": '$INITIAL_CHANNELS', "CLASS_WEIGHTS": '$CLASS_WEIGHTS'}'
# DESC='factor025_bsize128'

# PARAMS_LIST[${#PARAMS_LIST[@]}]=$PARAMS
# DESCRIPTIONS[${#DESCRIPTIONS[@]}]=$DESC

# PARAMS='{"N_CLASSES": '$N_CLASSES', "SR": '$SR', "BATCH_SIZE": 128, "LR": 0.001, "SHAPE": [257, '$WIDTH'], 
# "LL2_REG": 0.00, "WEIGHT_DECAY": 0.0001, "N_EPOCHS": '$N_EPOCHS', "FACTOR": 0.75, "PATIENCE":5, "MIN_LR":0.0001, "EPSILON": '$EPSILON', "ES_PATIENCE": '$ES_PATIENCE', "SAVE": '$SAVE', "ADD_TUNED": '$ADD_TUNED', "OVERSAMPLE": '$OVERSAMPLE', "MIN_DELTA": '$MIN_DELTA', "INITIAL_CHANNELS": '$INITIAL_CHANNELS', "CLASS_WEIGHTS": '$CLASS_WEIGHTS'}'
# DESC='factor075_bsize128'

# PARAMS_LIST[${#PARAMS_LIST[@]}]=$PARAMS
# DESCRIPTIONS[${#DESCRIPTIONS[@]}]=$DESC


# PARAMS='{"N_CLASSES": '$N_CLASSES', "SR": '$SR', "BATCH_SIZE": 64, "LR": 0.001, "SHAPE": [257, '$WIDTH'], 
# "LL2_REG": 0.00, "WEIGHT_DECAY": 0.0001, "N_EPOCHS": '$N_EPOCHS', "FACTOR": 0.5, "PATIENCE":5, "MIN_LR":0.0001, "EPSILON": '$EPSILON', "ES_PATIENCE": '$ES_PATIENCE', "SAVE": '$SAVE', "ADD_TUNED": '$ADD_TUNED', "OVERSAMPLE": '$OVERSAMPLE', "MIN_DELTA": '$MIN_DELTA', "INITIAL_CHANNELS": '$INITIAL_CHANNELS', "CLASS_WEIGHTS": '$CLASS_WEIGHTS'}'
# DESC='regular_regless'

# PARAMS_LIST[${#PARAMS_LIST[@]}]=$PARAMS
# DESCRIPTIONS[${#DESCRIPTIONS[@]}]=$DESC

# PARAMS='{"N_CLASSES": '$N_CLASSES', "SR": '$SR', "BATCH_SIZE": 64, "LR": 0.001, "SHAPE": [257, '$WIDTH'],
# "LL2_REG": 0, "WEIGHT_DECAY": 0.0001, "N_EPOCHS": '$N_EPOCHS', "FACTOR": 0.25, "PATIENCE":5, "MIN_LR":0.0001, "EPSILON": '$EPSILON', "ES_PATIENCE": '$ES_PATIENCE', "SAVE": '$SAVE', "ADD_TUNED": '$ADD_TUNED', "OVERSAMPLE": '$OVERSAMPLE', "MIN_DELTA": '$MIN_DELTA', "INITIAL_CHANNELS": '$INITIAL_CHANNELS', "CLASS_WEIGHTS": '$CLASS_WEIGHTS'}'
# DESC='factor025_regless'

# PARAMS_LIST[${#PARAMS_LIST[@]}]=$PARAMS
# DESCRIPTIONS[${#DESCRIPTIONS[@]}]=$DESC

# PARAMS='{"N_CLASSES": '$N_CLASSES', "SR": '$SR', "BATCH_SIZE": 64, "LR": 0.001, "SHAPE": [257, '$WIDTH'], 
# "LL2_REG": 0.000, "WEIGHT_DECAY": 0.0001, "N_EPOCHS": '$N_EPOCHS', "FACTOR": 0.75, "PATIENCE":5, "MIN_LR":0.0001, "EPSILON": '$EPSILON', "ES_PATIENCE": '$ES_PATIENCE', "SAVE": '$SAVE', "ADD_TUNED": '$ADD_TUNED', "OVERSAMPLE": '$OVERSAMPLE', "MIN_DELTA": '$MIN_DELTA', "INITIAL_CHANNELS": '$INITIAL_CHANNELS', "CLASS_WEIGHTS": '$CLASS_WEIGHTS'}'
# DESC='factor075_regless'

# PARAMS_LIST[${#PARAMS_LIST[@]}]=$PARAMS
# DESCRIPTIONS[${#DESCRIPTIONS[@]}]=$DESC

# PARAMS='{"N_CLASSES": '$N_CLASSES', "SR": '$SR', "BATCH_SIZE": 128, "LR": 0.001, "SHAPE": [257, '$WIDTH'], 
# "LL2_REG": 0.000, "WEIGHT_DECAY": 0.0001, "N_EPOCHS": '$N_EPOCHS', "FACTOR": 0.5, "PATIENCE":5, "MIN_LR":0.0001, "EPSILON": '$EPSILON', "ES_PATIENCE": '$ES_PATIENCE', "SAVE": '$SAVE', "ADD_TUNED": '$ADD_TUNED', "OVERSAMPLE": '$OVERSAMPLE', "MIN_DELTA": '$MIN_DELTA', "INITIAL_CHANNELS": '$INITIAL_CHANNELS', "CLASS_WEIGHTS": '$CLASS_WEIGHTS'}'
# DESC='regular_bsize128_regless'

# PARAMS_LIST[${#PARAMS_LIST[@]}]=$PARAMS
# DESCRIPTIONS[${#DESCRIPTIONS[@]}]=$DESC

# PARAMS='{"N_CLASSES": '$N_CLASSES', "SR": '$SR', "BATCH_SIZE": 128, "LR": 0.001, "SHAPE": [257, '$WIDTH'],
# "LL2_REG": 0.000, "WEIGHT_DECAY": 0.0001, "N_EPOCHS": '$N_EPOCHS', "FACTOR": 0.25, "PATIENCE":5, "MIN_LR":0.0001, "EPSILON": '$EPSILON', "ES_PATIENCE": '$ES_PATIENCE', "SAVE": '$SAVE', "ADD_TUNED": '$ADD_TUNED', "OVERSAMPLE": '$OVERSAMPLE', "MIN_DELTA": '$MIN_DELTA', "INITIAL_CHANNELS": '$INITIAL_CHANNELS', "CLASS_WEIGHTS": '$CLASS_WEIGHTS'}'
# DESC='factor025_bsize128_regless'

# PARAMS_LIST[${#PARAMS_LIST[@]}]=$PARAMS
# DESCRIPTIONS[${#DESCRIPTIONS[@]}]=$DESC

# PARAMS='{"N_CLASSES": '$N_CLASSES', "SR": '$SR', "BATCH_SIZE": 128, "LR": 0.001, "SHAPE": [257, '$WIDTH'], 
# "LL2_REG": 0.000, "WEIGHT_DECAY": 0.0001, "N_EPOCHS": '$N_EPOCHS', "FACTOR": 0.75, "PATIENCE":5, "MIN_LR":0.0001, "EPSILON": '$EPSILON', "ES_PATIENCE": '$ES_PATIENCE', "SAVE": '$SAVE', "ADD_TUNED": '$ADD_TUNED', "OVERSAMPLE": '$OVERSAMPLE', "MIN_DELTA": '$MIN_DELTA', "INITIAL_CHANNELS": '$INITIAL_CHANNELS', "CLASS_WEIGHTS": '$CLASS_WEIGHTS'}'
# DESC='factor075_bsize128_regless'

# PARAMS_LIST[${#PARAMS_LIST[@]}]=$PARAMS
# DESCRIPTIONS[${#DESCRIPTIONS[@]}]=$DESC

# PARAMS='{"N_CLASSES": '$N_CLASSES', "SR": '$SR', "BATCH_SIZE": 128, "LR": 0.001, "SHAPE": [257, '$WIDTH'], 
# "LL2_REG": 0.000, "WEIGHT_DECAY": 0.0001, "N_EPOCHS": '$N_EPOCHS', "FACTOR": 0.75, "PATIENCE":5, "MIN_LR":0.0001, "EPSILON": '$EPSILON', "ES_PATIENCE": '$ES_PATIENCE', "SAVE": '$SAVE', "ADD_TUNED": '$ADD_TUNED', "OVERSAMPLE": '$OVERSAMPLE', "MIN_DELTA": '$MIN_DELTA', "INITIAL_CHANNELS": '$INITIAL_CHANNELS', "CLASS_WEIGHTS": '$CLASS_WEIGHTS'}'
# DESC='factor075_bsize128_regmore'

# PARAMS_LIST[${#PARAMS_LIST[@]}]=$PARAMS
# DESCRIPTIONS[${#DESCRIPTIONS[@]}]=$DESC

# PARAMS='{"N_CLASSES": '$N_CLASSES', "SR": '$SR', "BATCH_SIZE": 128, "LR": 0.001, "SHAPE": [257, '$WIDTH'], 
# "LL2_REG": 0.000, "WEIGHT_DECAY": 0.0001, "N_EPOCHS": '$N_EPOCHS', "FACTOR": 0.75, "PATIENCE":3, "MIN_LR":0.0001, "EPSILON": '$EPSILON', "ES_PATIENCE": '$ES_PATIENCE', "SAVE": '$SAVE', "ADD_TUNED": '$ADD_TUNED', "OVERSAMPLE": '$OVERSAMPLE', "MIN_DELTA": '$MIN_DELTA', "INITIAL_CHANNELS": '$INITIAL_CHANNELS', "CLASS_WEIGHTS": '$CLASS_WEIGHTS'}'
# DESC='factor075_bsize128_regmore_lesspatience'

# PARAMS_LIST[${#PARAMS_LIST[@]}]=$PARAMS
# DESCRIPTIONS[${#DESCRIPTIONS[@]}]=$DESC

# PARAMS='{"N_CLASSES": '$N_CLASSES', "SR": '$SR', "BATCH_SIZE": 128, "LR": 0.001, "SHAPE": [257, '$WIDTH'], 
# "LL2_REG": 0.000, "WEIGHT_DECAY": 0.0001, "N_EPOCHS": '$N_EPOCHS', "FACTOR": 0.75, "PATIENCE":8, "MIN_LR":0.0001, "EPSILON": '$EPSILON', "ES_PATIENCE": '$ES_PATIENCE', "SAVE": '$SAVE', "ADD_TUNED": '$ADD_TUNED', "OVERSAMPLE": '$OVERSAMPLE', "MIN_DELTA": '$MIN_DELTA', "INITIAL_CHANNELS": '$INITIAL_CHANNELS', "CLASS_WEIGHTS": '$CLASS_WEIGHTS'}'
# DESC='factor075_bsize128_regmore_morepatience'

# PARAMS_LIST[${#PARAMS_LIST[@]}]=$PARAMS
# DESCRIPTIONS[${#DESCRIPTIONS[@]}]=$DESC

# PARAMS='{"N_CLASSES": '$N_CLASSES', "SR": '$SR', "BATCH_SIZE": 128, "LR": 0.001, "SHAPE": [257, '$WIDTH'], 
# "LL2_REG": 0.000, "WEIGHT_DECAY": 0.0001, "N_EPOCHS": '$N_EPOCHS', "FACTOR": 0.75, "PATIENCE":8, "MIN_LR":0.0001, "EPSILON": '$EPSILON', "ES_PATIENCE": '$ES_PATIENCE', "SAVE": '$SAVE', "ADD_TUNED": '$ADD_TUNED', "OVERSAMPLE": '$OVERSAMPLE', "MIN_DELTA": '$MIN_DELTA', "INITIAL_CHANNELS": '$INITIAL_CHANNELS', "CLASS_WEIGHTS": '$CLASS_WEIGHTS'}'
# DESC='factor075_bsize128_regless5_morepatience'

# PARAMS_LIST[${#PARAMS_LIST[@]}]=$PARAMS
# DESCRIPTIONS[${#DESCRIPTIONS[@]}]=$DESC

# PARAMS='{"N_CLASSES": '$N_CLASSES', "SR": '$SR', "BATCH_SIZE": 128, "LR": 0.001, "SHAPE": [257, '$WIDTH'], 
# "LL2_REG": 0.000, "WEIGHT_DECAY": 0.0001, "N_EPOCHS": '$N_EPOCHS', "FACTOR": 0.75, "PATIENCE":8, "MIN_LR":0.0001, "EPSILON": '$EPSILON', "ES_PATIENCE": '$ES_PATIENCE', "SAVE": '$SAVE', "ADD_TUNED": '$ADD_TUNED', "OVERSAMPLE": '$OVERSAMPLE', "MIN_DELTA": '$MIN_DELTA', "INITIAL_CHANNELS": '$INITIAL_CHANNELS', "CLASS_WEIGHTS": '$CLASS_WEIGHTS'}'
# DESC='factor075_bsize128_regless15_morepatience'

# PARAMS_LIST[${#PARAMS_LIST[@]}]=$PARAMS
# DESCRIPTIONS[${#DESCRIPTIONS[@]}]=$DESC

# PARAMS='{"N_CLASSES": '$N_CLASSES', "SR": '$SR', "BATCH_SIZE": 128, "LR": 0.01, "SHAPE": [257, '$WIDTH'], 
# "LL2_REG": 0.000, "WEIGHT_DECAY": 0.0001, "N_EPOCHS": '$N_EPOCHS', "FACTOR": 0.75, "PATIENCE":4, "MIN_LR":0.0001, "EPSILON": '$EPSILON', "ES_PATIENCE": '$ES_PATIENCE', "SAVE": '$SAVE', "ADD_TUNED": '$ADD_TUNED', "OVERSAMPLE": '$OVERSAMPLE', "MIN_DELTA": '$MIN_DELTA', "INITIAL_CHANNELS": '$INITIAL_CHANNELS', "CLASS_WEIGHTS": '$CLASS_WEIGHTS'}'
# DESC='001_factor075_bsize128_regless15_morepatience'

# PARAMS_LIST[${#PARAMS_LIST[@]}]=$PARAMS
# DESCRIPTIONS[${#DESCRIPTIONS[@]}]=$DESC

# #########################
# # rarely changeable

# MASS_REGION=us-central1
# MASS_BUCKET_NAME=tf_learn_pattern_detection
# MASS_LOCAL_TRAIN_FILE=../../../data/datasets/$MASS_FILE_NAME
# MASS_PACKAGE_PATH=/Users/alirachidi/Documents/Sonavi_Labs/ObjectDetection/trainers/main/$MASS_FOLDER/$MASS_MODEL
# MASS_TRAIN_FILE=gs://$MASS_BUCKET_NAME/datasets/$MASS_FILE_NAME

# echo "Train File: " $MASS_TRAIN_FILE


# ############################
# # defining job names

# let "INDEX = $((${#MASS_FILE_NAME}-4))"
# NAME=${MASS_FILE_NAME:0:$INDEX}

# declare -A MASS_JOB_NAMES=()

# for ((i = 0; i < ${#DESCRIPTIONS[@]}; ++i));
# do
#     MASS_JOB_NAMES[${#MASS_JOB_NAMES[@]}]=$MASS_MODEL'___'$(date +%m_%H%M)'___'$NAME'___'"${DESCRIPTIONS[$i]}"'___'$SOURCE'__'$EXPERIMENT_NB'__'$i
# done

# ######################

# printf "\n#### Starting ${#PARAMS_LIST[@]} experiments ... #####\n\n"

# for ((i = 0; i < ${#PARAMS_LIST[@]}; ++i));
# do
#     printf "###### Preparing Job $i! ######\n\n"
#     MASS_JOB_DIR=gs://$MASS_BUCKET_NAME/$MASS_JOB_CAT/$MASS_MODEL/$EXPERIMENT/${MASS_JOB_NAMES[$i]}  
#     #echo "Description of Job $i: ${DESCRIPTIONS[$i]}"
#     echo "Name of Job $i: ${MASS_JOB_NAMES[$i]}"
#     echo "Params of Job $i: ${PARAMS_LIST[$i]}" 
#     sleep 0.5
#     if [ "$LOCAL" = false ]; then
#         week=$(date +%U)
#         year=$(date +%Y)
#         echo "File Number: $MASS_MODULE_NAME" >> 'records/'$week'_'$year'_records.txt'
#         echo "Params of Job $i: ${PARAMS_LIST[$i]}" >> 'records/'$week'_'$year'_records.txt'
#         echo -e "Name of Job $i: ${MASS_JOB_NAMES[$i]} \n" >> 'records/'$week'_'$year'_records.txt'
#         gcloud ai-platform jobs submit training "${MASS_JOB_NAMES[$i]}" --job-dir $MASS_JOB_DIR --module-name $MASS_MODULE_NAME --package-path $MASS_PACKAGE_PATH --region $MASS_REGION --config=./cloudml-gpu.yaml -- --train-file $MASS_TRAIN_FILE --params "${PARAMS_LIST[$i]}"
#     else
#         MASS_LOCAL_JOB_DIR=../../../cache/${MASS_JOB_NAMES[$i]}
#         gcloud ai-platform local train --module-name $MASS_MODULE_NAME --package-path $MASS_PACKAGE_PATH -- --train-file $MASS_LOCAL_TRAIN_FILE --job-dir $MASS_LOCAL_JOB_DIR --params "${PARAMS_LIST[$i]}"
#     fi
#     printf "\n###### Job $i was launched! ######\n\n"
# done
# printf "All jobs were launched - check logs at gs://$MASS_BUCKET_NAME/$MASS_JOB_CAT/$MASS_MODEL/$EXPERIMENT/ !"