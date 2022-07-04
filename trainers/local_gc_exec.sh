export TESTING=0 # allows for debugging by picking a smaller dataset, lower number of epochs, etc
export MODE=cw #folder indicaion: trainers/main (=> pneumonia) vs trainers/cw (=> crackles and wheezes)
export TRAIN_NUMBER=222
export DESCRIPTION="kfold with all data"  
export MODULE_NAME=train$TRAIN_NUMBER
export OUTPUT_FILE=$MODE/job_outputs/${TRAIN_NUMBER}.out # log file: destination for .out file if nohup is used (example: pneumonia/job_outputs/1.out)

echo "Mode: " $MODE
echo "Testing: " $TESTING
echo "File: " $MODULE_NAME
echo "Description: " $DESCRIPTION
echo "Output File: " $OUTPUT_FILE

####### 3 inputs:
# 1) training file passed as module
# 2) testing: 0 or 1
# 3): description: some description of the job(s) about to be done

export DEVICE=0
# CUDA_VISIBLE_DEVICES=$DEVICE nohup python -m $MODE.models.$MODULE_NAME --testing "$TESTING"  --description "$DESCRIPTION" --mode "$MODE" > $OUTPUT_FILE & 

#### non-nohup
CUDA_VISIBLE_DEVICES=$DEVICE /home/alirachidi/anaconda3/envs/LungSoundClass/bin/python3 -m $MODE.models.$MODULE_NAME --testing "$TESTING"  --description "$DESCRIPTION"  --mode "$MODE"