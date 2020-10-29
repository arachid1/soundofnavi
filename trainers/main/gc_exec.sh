echo "Category: " $JOB_CAT
echo "Dataset: " $FILE_NAME
echo "File: " $MODULE_NAME
echo "Description: " $DESCRIPTION
echo "Parameters: " $PARAMS

sleep 2

echo "Config: " $YAML_CONFIG

week=$(date +%U)
year=$(date +%Y)
echo "File Number: $MODULE_NAME" >> 'records/'$week'_'$year'_records.txt'
echo "Params of Job: $PARAMS" >> 'records/'$week'_'$year'_records.txt'
echo -e "Name of Job: $JOB_NAME\n" >> 'records/'$week'_'$year'_records.txt'

gcloud ai-platform jobs submit training $JOB_NAME --job-dir $JOB_DIR --module-name $MODULE_NAME --package-path $PACKAGE_PATH --region $REGION --config=./cloudml-gpu.yaml -- --train-file $TRAIN_FILE --params "$PARAMS"

printf "\n See logs at $JOB_DIR/logs/"