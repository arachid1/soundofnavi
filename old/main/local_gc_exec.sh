echo "Category: " $JOB_CAT
echo "Dataset: " $FILE_NAME
echo "File: " $MODULE_NAME
echo "Description: " $DESCRIPTION

python -m models.$MODULE_NAME --train-file $LOCAL_TRAIN_FILE --job-dir $LOCAL_JOB_DIR --params "$PARAMS" --wav-params "$WAV_PARAMS" --spec-params "$SPEC_PARAMS"
# # /home/alirachidi/.conda/envs/tf1/bin/python -m models.$MODULE_NAME --train-file $LOCAL_TRAIN_FILE --job-dir $LOCAL_JOB_DIR --params "$PARAMS"
printf "\n See logs at $LOCAL_JOB_DIR/logs/"