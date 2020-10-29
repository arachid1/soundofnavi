echo "Category: " $JOB_CAT
echo "Dataset: " $FILE_NAME
echo "File: " $MODULE_NAME
echo "Description: " $DESCRIPTION

gcloud ai-platform local train --module-name $MODULE_NAME --package-path $PACKAGE_PATH -- --train-file $LOCAL_TRAIN_FILE --job-dir $LOCAL_JOB_DIR --params "$PARAMS"
#/opt/anaconda3/envs/ObjectDetection/bin/python -m $MODULE_NAME -h --train-file $LOCAL_TRAIN_FILE --job-dir $LOCAL_JOB_DIR
#/opt/anaconda3/envs/nightly/bin/python -m $MODULE_NAME --train-file $LOCAL_TRAIN_FILE --job-dir $LOCAL_JOB_DIR
printf "\n See logs at $LOCAL_JOB_DIR/logs/"