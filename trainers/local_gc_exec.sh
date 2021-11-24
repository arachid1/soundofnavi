export TESTING=1 # allows for debugging by picking a smaller dataset/lower number of epochs
export MODE=main # folder indication: trainers/main (=> pneumonia) vs trainers/cw for crackles and wheezes
export TRAIN_NUMBER=3
export DESCRIPTION="some description"
export MODULE_NAME=train$TRAIN_NUMBER
export OUTPUT_FILE=$MODE/job_outputs/${TRAIN_NUMBER}.out # destination for .out file if nohup is used (example: pneumonia/job_outputs/1.out)

echo "Mode: " $MODE
echo "Testing: " $TESTING
echo "File: " $MODULE_NAME
echo "Description: " $DESCRIPTION
echo "Output File: " $OUTPUT_FILE

# 3 inputs:
# 1) training file passed as module
# 2) testing: 0 or 1
# 3): description: some description of the job(s) about to be done

# nohup
# CUDA_VISIBLE_DEVICES=0 nohup python -m $MODE.models.$MODULE_NAME --testing "$TESTING"  --description "$DESCRIPTION" > $OUTPUT_FILE & 

# non-nohup
# CUDA_VISIBLE_DEVICES=-1  python -m $MODE.models.$MODULE_NAME --testing "$TESTING"  --description "$DESCRIPTION" 
CUDA_VISIBLE_DEVICES=0  python -m $MODE.models.$MODULE_NAME --testing "$TESTING"  --description "$DESCRIPTION" 