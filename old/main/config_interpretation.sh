#!/usr/bin/env bash

export BUCKET_NAME=tf_learn_pattern_detection

######################
#  parameters that are often changed

export FOLDER=models
export JOB_CAT=quantization #LOG OR MEL

SR=8000
#perch_sw_log_param_v2_8000.pkl
# perch_sw_log_param_v2_$SR.pkl
export FILE_NAME="all_sw_log_preprocessed_v2_param_v2_$SR.pkl"
export MODEL=conv #LSTM OR CONV
export TRAIN_NUMBER=9999
export DESCRIPTION=attempt_quantization

export ID=88
export SAVED_MODEL_DIR=gs://$BUCKET_NAME/$JOB_CAT/$ID/
# export TFLITE_NAME=model.tflite


export PARAMS='{"SAVED_MODEL_DIR": "'$SAVED_MODEL_DIR'"}'

######################
export FILE=$(echo $FILE_NAME| cut -d'.' -f 1)
export DETAILS='___'$DESCRIPTION'___'$TRAIN_NUMBER

export JOB_NAME=$MODEL'___'$(date +%m_%H%M)'___'$FILE$DETAILS
export JOB_DIR=gs://$BUCKET_NAME/$JOB_CAT/$JOB_NAME
export TRAIN_FILE=gs://$BUCKET_NAME/datasets/$FILE_NAME

export MODULE_NAME=$MODEL.train$TRAIN_NUMBER
export PACKAGE_PATH=/Users/alirachidi/Documents/Sonavi_Labs/classification_algorithm/trainers/main/$FOLDER/$MODEL

export LOCAL_TRAIN_FILE=../../../data/datasets/$FILE_NAME
export LOCAL_JOB_DIR=../../../cache/datasets/$JOB_NAME

export REGION=us-central1


echo "File: " $MODULE_NAME
echo "Category: " $JOB_CAT
echo "Description: " $DESCRIPTION
echo "Dataset: " $FILE_NAME
echo "Params:" $PARAMS


read confirmation
