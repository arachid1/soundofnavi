if [ $# -eq 0 ]
then
    echo "No arguments supplied"
    #export NAME=conv2d___08_1402___icbhi_sw_log_spec_v1_8000___epsilon01___106
    #export LOG_DIR=gs://$BUCKET_NAME/$JOB_CAT/$MODEL/$NAME/logs/
    export LOG_DIR=gs://tf_learn_pattern_detection/log/conv/exp_222_1/
    lsof -i :8000 | tail -n +2 | awk '{system("kill -s 9 " $2)}' &> /dev/null;
    tensorboard --logdir=$LOG_DIR --port 8000 --reload_interval=5
    open -a "Google Chrome" localhost:8000
else
    export NAME=$1
    SUB='mel'
    [[ "$NAME" == *"$SUB"* ]] && CAT=mel || CAT=coch
    export LOG_DIR=gs://$BUCKET_NAME/$CAT/$MODEL/$NAME/logs
    if [ -z ${2} ]; 
    then 
        echo "Port is unset"; 
        open -a "Google Chrome" http://localhost:8000
        lsof -i :8000 | tail -n +2 | awk '{system("kill -s 9 " $2)}' &> /dev/null;
        tensorboard --logdir=$LOG_DIR --port 8000 --reload_interval=5
    else 
        echo "Port is set to $2"; 
        lsof -i :$2 | tail -n +2 | awk '{system("kill -s 9 " $2)}' &> /dev/null;
        open -a "Google Chrome" http://localhost:$2
        tensorboard --logdir=$LOG_DIR --port $2 --reload_interval=5
    fi
fi

echo "Log Directory: $LOG_DIR"

# GCLOUD VIEW

# LOCAL VIEW
#tensorboard --logdir=$LOCAL_JOB_DIR/logs --port 8000 --reload_interval=5
#tensorboard --logdir=../../../cache/tf_learn_pattern_detection___conv2d___20200723_171537___data_slidingwindow_log_spectrogram___template___100/logs --port 8000 --reload_interval=5
