#!/usr/bin/env bash

# - No -> Is there a current training?
# - - No -> Do nothing
# - - Yes -> create  job_running.dat and write the current job
# - Yes -> Is there a current training?
# - - Yes -> Is it the same?
# - - - Yes -> do nothing
# - - - No -> SEND NOTIFICATION and write new current job to job_running.dat
# - - No -> SEND NOTIFICATION and update delete file



output=$(gcloud ai-platform jobs list --filter="STATUS=RUNNING OR PREPARING")
while read -r line; do
    process "$line"
done <<< "$output"
stringarray=($output)
job_name=${stringarray[3]}
file_path="temp/current_job.dat"



# did current_job.dat exist? 
if [ ! -f $file_path ]
then # No
  echo "No training found 5 minutes ago. Let's see now..."
  if [ -z "$output" ] # Is there a current training?
  then # No -> Do nothing
    echo "No new training. "
    true
  else # Yes -> create job_running.dat and write the current job
    echo "Found new training: \"$job_name\" "
    # echo $file_path
    echo "$job_name" > "$file_path"
  fi
else # Yes
#   old_job=`cat \"$file_path\"`
  old_job=$(<$file_path)
  echo "A Job was running 5 minutes ago. The name is \"$old_job\""
  if [ -z "$output" ] # Is there a current training?
  then  #No -> SEND NOTIFICATION and delete file
    afplay /System/Library/Sounds/Hero.aiff
    echo "The job stopped running. Notification Sent!"
    rm $file_path
  else # Yes
    job_name="$job_name"
    old_job="$old_job"
    if [[ "$job_name" == "$old_job" ]] # Is it the same?
    then # Yes -> do nothing
        echo "The same job is running. Let's check again later!"
    else # No -> SEND NOTIFICATION and write new current job to job_running.dat
        afplay /System/Library/Sounds/Hero.aiff
        echo "A different job is/has started since the last check. "
        echo "The new job name is \"$job_name\". Records were updated. Notification Sent!"
        rm $file_path
        echo $job_name >> $file_path
    fi
  fi
fi
