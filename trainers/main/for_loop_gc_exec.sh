#!/usr/bin/env bash
BUCKET_NAME=tf_learn_pattern_detection

FOLDER=models
JOB_CAT=log #LOG OR MEL
MODEL=conv
LOCAL=false

FILE_NUMBER=3000

SR=8000
PREPROCESSED=v2
PARAM=v2
AUGM=v0
FILE_NAME="all_sw_${JOB_CAT}_preprocessed_${PREPROCESSED}_param_${PARAM}_augm_${AUGM}_$SR.pkl"

N_EPOCHS=100
N_CLASSES=2
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
ES_PATIENCE=8
MIN_DELTA=0.01
INITIAL_CHANNELS=3

CLASS_WEIGHTS=1
SAVE=0
ADD_TUNED=0
OVERSAMPLE=0

TRAIN_FILE=gs://$BUCKET_NAME/datasets/$FILE_NAME

PACKAGE_PATH=/Users/alirachidi/Documents/Sonavi_Labs/classification_algorithm/trainers/main/$FOLDER/$MODEL

LOCAL_TRAIN_FILE=../../../data/datasets/$FILE_NAME
LOCAL_JOB_DIR=../../../cache/datasets/$JOB_NAME

REGION=us-central1

##### 

KERNEL_SIZES=(5 6 7)
POOL_SIZES=(2 3 4)
CHANNELS=(4 8 12 16 24 32 48 64 96 128 192 256 384 512 768)
DROPOUTS=(0.1)
PADDINGS=(0) #same
DENSE_LAYERS=(0 4 8 32 64)

HYPER_PARAMS='{"N_CLASSES": '$N_CLASSES', "SR": '$SR', "BATCH_SIZE": '$BATCH_SIZE', "LR": '$LR', "SHAPE": ['$HEIGHT', '$WIDTH'], "LL2_REG": '$LL2_REG', "WEIGHT_DECAY": '$WEIGHT_DECAY', "N_EPOCHS": '$N_EPOCHS', "FACTOR": '$FACTOR', "PATIENCE":'$PATIENCE', "MIN_LR":'$MIN_LR', "EPSILON": '$EPSILON', "ES_PATIENCE": '$ES_PATIENCE', "SAVE": '$SAVE', "ADD_TUNED": '$ADD_TUNED', "OVERSAMPLE": '$OVERSAMPLE', "MIN_DELTA": '$MIN_DELTA', "INITIAL_CHANNELS": '$INITIAL_CHANNELS', "CLASS_WEIGHTS": '$CLASS_WEIGHTS'}'

# find the next id 
id=$(< "records.csv" wc -l)
id=$(($id+1))
echo "First id: $id"
echo "Len: ${#CHANNELS[@]}"
count=0

for ((i = 0; i < ${#KERNEL_SIZES[@]}; ++i));
do
    for ((j = 0; j < ${#POOL_SIZES[@]}; ++j));
    do
        for ((l = 0; l < ${#DROPOUTS[@]}; ++l));
        do
            for ((m = 0; m < ${#PADDINGS[@]}; ++m));
            do 
                for ((n = 0; n < ${#DENSE_LAYERS[@]}; ++n));
                do
                    for ((k = 0; k < ${#CHANNELS[@]}; ++k));
                    do
                        # set a limit on number of jobs
                        if [ $count -eq 37 ] #TODO: FIXXXXX
                        then
                            exit 2
                        fi
                        # making sure channels accessed are within bound
                        if [ "$((k+6))" -gt "${#CHANNELS[@]}" ]
                            then 
                                echo "k = $k is too big; break"
                                break
                        fi 
                        # incrementing count because it's a valid job
                        count=$((count+1))
                        echo "count: $count"
                        # decide which job will we start from
                        if [ $count -ge $id ] 
                        then
                            echo "count has exceeded id so we are back on the pipeline"
                            # if [ "$((k+6))" -gt "${#CHANNELS[@]}" ]
                            # then 
                            #     echo "k = $k is too big; break"
                            #     break
                            # else
                            
                            # defining the architecture parameters
                            SLICED_CHANNELS='['${CHANNELS[k]}', '${CHANNELS[k+1]}', '${CHANNELS[k+2]}', '${CHANNELS[k+3]}', '${CHANNELS[k+4]}', '${CHANNELS[k+5]}']'
                            ARCH_PARAMS='{"KERNEL_SIZE": '${KERNEL_SIZES[$i]}', "POOL_SIZE": '${POOL_SIZES[$j]}', "DROPOUT": '${DROPOUTS[$l]}', "PADDING": '${PADDINGS[$m]}', "DENSE_LAYER": '${DENSE_LAYERS[$n]}', "CHANNELS": '$SLICED_CHANNELS'}'
                            echo "###### Preparing Job with id = $id ######"
                            FILE=($(echo $FILE_NAME| cut -d'.' -f 1))
                            DESCRIPTION="id_$id"
                            DETAILS=('___'${DESCRIPTION}'___'${FILE_NUMBER})
                            NAME="$MODEL"___$(date +%m_%H%M)___"$FILE$DETAILS"
                            JOB_DIR=gs://$BUCKET_NAME/$JOB_CAT/$MODEL/$NAME
                            MODULE_NAME=$MODEL.train${FILE_NUMBER}
                            echo "Name of Job: $NAME"
                            echo "Params of Job: $ARCH_PARAMS"
                            if [ "$LOCAL" = false ]; 
                            then
                                week=$(date +%U)
                                year=$(date +%Y)
                                echo "File Number: $MODULE_NAME" >> 'records/'$week'_'$year'_records.txt'
                                echo "Params of Job: $PARAMS" >> 'records/'$week'_'$year'_records.txt' #include arch params
                                echo -e "Name of Job: $NAME\n" >> 'records/'$week'_'$year'_records.txt'
                                gcloud ai-platform jobs submit training $NAME --job-dir $JOB_DIR --module-name $MODULE_NAME --package-path $PACKAGE_PATH --region $REGION --config=./cloudml-gpu.yaml -- --train-file $TRAIN_FILE --hyper-params "$HYPER_PARAMS" --arch-params "$ARCH_PARAMS" 
                            else
                                LOCAL_JOB_DIR=../../../cache/${JOB_NAMES[$i]}
                                gcloud ai-platform local train --module-name $MODULE_NAME --package-path $PACKAGE_PATH -- --train-file $LOCAL_TRAIN_FILE --job-dir $LOCAL_JOB_DIR --hyper-params "$HYPER_PARAMS" --arch-params "$ARCH_PARAMS"
                            fi
                            id=$((id+1))    

                            # fi  
                        else
                            continue 
                        fi
                    done
                done
            done
        done
    done
done

# for kernel_size in $KERNEL_SIZES
# do
#     for pool_size in $POOL_SIZES
#     do
#         echo $kernel_size
#         echo $pool_size
#     done
# done

# init other params
# init for loop params

# for loop of params:
#     build params from both (how to merge 2 dict or just build one inside the for loop)
#     send job (give it an id)

# final_id=0
# count=0
# printf '%s\n' "Kernel Size" "Pool Size" "Dropout" | paste -s -d ' ' >> records.csv
# printf '%s\n' "4" "5" "0.1" | paste -s -d ' ' >> records.csv
# printf "Id,kernel_size,pool_size,dropout,padding,dense_layer,channels\n" >> records.csv

# while IFS=, read -r id
# do
#     echo "$id"
#     final_id=$id
# done < records.csv

# printf "9898,5,4,0.1,same,128,[23, 545, 25, 234, 23]" >> records.csv 

# echo ${matrix[1,0]}

# (cat records.csv ; echo) | while IFS=',' read -r id kernel_size pool_size dropout padding dense_layer channels_start channels_end
# do
#     echo $id
#     final_id=$id
#     # # [ $dropout -eq ${DROPOUTS[$l]} ] && echo "They're equal!"
#     # # (( $(echo "$dropout=${DROPOUTS[$l]}" |bc -l) )) && echo "They're equal!"
#     # if [ $kernel_size -eq ${KERNEL_SIZES[$i]} ] && [ $pool_size -eq ${POOL_SIZES[$j]} ] && [ $dropout -eq ${DROPOUTS[$l]} ] && [ $padding -eq ${PADDINGS[$m]} ] && [ $dense_layer -eq ${DENSE_LAYERS[$n]} ] && [ $channels -eq $SLICED_CHANNELS] ; then
#     #     echo "Found duplicate!"
#     #     $DUPLICATE=true
#     # fi
#     count=$((count+1))
# done
# echo $count
# cat records.csv | python -c "import csv; import sys; print(sum(1 for i in csv.reader(sys.stdin)))"
# DUPLICATE=false