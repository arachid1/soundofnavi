export TESTING=0 # activates an options where a smaller dataset is picked for debugging purposes
export MODE=main # main (=> pneumonia) vs cw for crackles and wheezes
export TRAIN_NUMBER=2419
export DESCRIPTION="other_new"
export MODULE_NAME=train$TRAIN_NUMBER
export OUTPUT_FILE=$MODE/job_outputs/${TRAIN_NUMBER}_2.out

echo "Mode: " $MODE
echo "Testing: " $TESTING
echo "File: " $MODULE_NAME
echo "Description: " $DESCRIPTION
echo "Output File: " $OUTPUT_FILE

# CUDA_VISIBLE_DEVICES=0 nohup python -m $MODE.models.$MODULE_NAME --testing "$TESTING"  --description "$DESCRIPTION" > $OUTPUT_FILE & 
# CUDA_VISIBLE_DEVICES=-1  python -m $MODE.models.$MODULE_NAME --testing "$TESTING"  --description "$DESCRIPTION" 
CUDA_VISIBLE_DEVICES=0  python -m $MODE.models.$MODULE_NAME --testing "$TESTING"  --description "$DESCRIPTION" 

#--mode "$MODE"
# --train-file $LOCAL_TRAIN_FILE --job-dir $LOCAL_JOB_DIR --params "$PARAMS" --wav-params "$WAV_PARAMS" --spec-params "$SPEC_PARAMS"
# python -m models.$MODULE_NAME --train-file $LOCAL_TRAIN_FILE --job-dir $LOCAL_JOB_DIR --params "$PARAMS" --wav-params "$WAV_PARAMS" --spec-params "$SPEC_PARAMS"
# # /home/alirachidi/.conda/envs/tf1/bin/python -m models.$MODULE_NAME --train-file $LOCAL_TRAIN_FILE --job-dir $LOCAL_JOB_DIR --params "$PARAMS"
# printf "\n See logs at $LOCAL_JOB_DIR/logs/"