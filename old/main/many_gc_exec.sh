#!/usr/bin/env bash
BUCKET_NAME=tf_learn_pattern_detection

##### log or mel

FOLDER=models
JOB_CAT=log #LOG OR MEL
MODEL=conv
LOCAL=false

#perch_sw_log_param_v2_8000.pkl
# "all_sw_log_preprocessed_v1_param_v2_$SR.pkl"
# perch_sw_log_param_v2_$SR.pkl
SR=8000
FILE_NAME="all_sw_log_preprocessed_v2_param_v2_$SR.pkl"

N_EPOCHS=100
N_CLASSES=1
SR=8000
EPSILON=1e-7 # default: 1e-7

[ "$JOB_CAT" = log ] && HEIGHT=257 || HEIGHT=128 
[ "$SR" = 8000 ] && WIDTH=251 || WIDTH=126 

BATCH_SIZE=64
WEIGHT_DECAY=1e-4 
LL2_REG=0 # default: wd=1e-4
LR=1e-4 # default: 1e-3
MIN_LR=1e-5 # default: 1e-4
FACTOR=0.5 # default: 0.5
PATIENCE=5 # default: 5
ES_PATIENCE=12
MIN_DELTA=0.01
INITIAL_CHANNELS=3

CLASS_WEIGHTS=0
SAVE=0
ADD_TUNED=0
OVERSAMPLE=1

TRAIN_FILE=gs://$BUCKET_NAME/datasets/$FILE_NAME

PACKAGE_PATH=/Users/alirachidi/Documents/Sonavi_Labs/classification_algorithm/trainers/main/$FOLDER/$MODEL

LOCAL_TRAIN_FILE=../../../data/datasets/$FILE_NAME
LOCAL_JOB_DIR=../../../cache/datasets/$JOB_NAME

REGION=us-central1

FILE_NUMBERS=(236 237 258 259)
DESCRIPTIONS=("softmax_crackles" "softmax_wheezes"  "softmax_none_vs_rest" "softmax_both_vs_rest")


PARAMS='{"N_CLASSES": '$N_CLASSES', "SR": '$SR', "BATCH_SIZE": '$BATCH_SIZE', "LR": '$LR', "SHAPE": ['$HEIGHT', '$WIDTH'], 
"LL2_REG": '$LL2_REG', "WEIGHT_DECAY": '$WEIGHT_DECAY', "N_EPOCHS": '$N_EPOCHS', "FACTOR": '$FACTOR', "PATIENCE":'$PATIENCE', "MIN_LR":'$MIN_LR', 
"EPSILON": '$EPSILON', "ES_PATIENCE": '$ES_PATIENCE', "SAVE": '$SAVE', "ADD_TUNED": '$ADD_TUNED', "OVERSAMPLE": '$OVERSAMPLE', "MIN_DELTA": '$MIN_DELTA', "INITIAL_CHANNELS": '$INITIAL_CHANNELS', "CLASS_WEIGHTS": '$CLASS_WEIGHTS'}'
# echo "Params: ${PARAMS}" 

echo "Category: " $JOB_CAT
echo "Dataset: " $FILE_NAME
echo "File Numbers: ${FILE_NUMBERS[*]}"
echo "Descriptions: ${DESCRIPTIONS[*]}"

read confirmation

echo "Learning Rate: $LR"
echo "Minimum Lr: $MIN_LR"
echo "Weight Decay: $WEIGHT_DECAY"
echo "LL2 Regulariziaiton: $LL2_REG"

read confirmation

echo "Save Images & Audios: $SAVE"
echo "Add Tuned Data: $ADD_TUNED"
echo "Oversample Minority Classes: $OVERSAMPLE"

read confirmation


printf "\n#### Starting ${#FILE_NUMBERS[@]} scripts ... #####\n\n"

for ((i = 0; i < ${#FILE_NUMBERS[@]}; ++i));
do
    echo "###### Preparing Job ${FILE_NUMBERS[$i]} ######"
    FILE=($(echo $FILE_NAME| cut -d'.' -f 1))
    DETAILS=('___'${DESCRIPTIONS[$i]}'___'${FILE_NUMBERS[$i]})
    NAME="$MODEL"___$(date +%m_%H%M)___"$FILE$DETAILS"
    JOB_DIR=gs://$BUCKET_NAME/$JOB_CAT/$MODEL/$NAME
    MODULE_NAME=$MODEL.train${FILE_NUMBERS[$i]}
    # echo $JOB_DIR
    echo "Name of Job ${FILE_NUMBERS[$i]}: $NAME"
    sleep 0.5
    if [ "$LOCAL" = false ]; then
        week=$(date +%U)
        year=$(date +%Y)
        echo "File Number: $MODULE_NAME" >> 'records/'$week'_'$year'_records.txt'
        echo "Params of Job: $PARAMS" >> 'records/'$week'_'$year'_records.txt'
        echo -e "Name of Job: $NAME\n" >> 'records/'$week'_'$year'_records.txt'
        gcloud ai-platform jobs submit training $NAME --job-dir $JOB_DIR --module-name $MODULE_NAME --package-path $PACKAGE_PATH --region $REGION --config=./cloudml-gpu.yaml -- --train-file $TRAIN_FILE --params "$PARAMS"
    else
        LOCAL_JOB_DIR=../../../cache/${JOB_NAMES[$i]}
        gcloud ai-platform local train --module-name $MODULE_NAME --package-path $PACKAGE_PATH -- --train-file $LOCAL_TRAIN_FILE --job-dir $LOCAL_JOB_DIR --params "$PARAMS"
    fi
    printf "check logs at gs://$BUCKET_NAME/$JOB_CAT/$MODEL/$NAME/ \n\n"
done