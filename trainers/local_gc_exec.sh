# allows for debugging by picking a smaller dataset, lower number of epochs, etc
export TESTING=0 
#folder indicaion: trainers/main (=> pneumonia) vs trainers/cw (=> crackles and wheezes)
export MODE=cw 
export TRAIN_NUMBER=221
export DESCRIPTION="kfold with all data"  
export MODULE_NAME=train$TRAIN_NUMBER
export OUTPUT_FILE=$MODE/job_outputs/${TRAIN_NUMBER}.out # log file: destination for .out file if nohup is used (example: pneumonia/job_outputs/1.out)

echo "________________________________________________________________"
echo "Mode: " $MODE
echo "Testing: " $TESTING
echo "File: " $MODULE_NAME
echo "Description: " $DESCRIPTION
echo "Output File: " $OUTPUT_FILE
echo "________________________________________________________________"

####### 3 inputs:
# 1) training file passed as module
# 2) testing: 0 or 1
# 3): description: some description of the job(s) about to be done

export DEVICE=-1

#### nohup 
# CUDA_VISIBLE_DEVICES=$DEVICE nohup python -m $MODE.models.$MODULE_NAME --testing "$TESTING"  --description "$DESCRIPTION" --mode "$MODE" > $OUTPUT_FILE & 

#### non-nohup
CUDA_VISIBLE_DEVICES=$DEVICE python -m $MODE.jobs.$MODULE_NAME --testing "$TESTING"  --description "$DESCRIPTION"  --mode "$MODE"