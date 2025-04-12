#!/bin/bash

export HF_HOME=/data/hf_cache/
export TOTAL_PROCS=32
export OUTPUT_DIR="/data/joel/prepared/realnewslike/"
mkdir -p $OUTPUT_DIR

for (( i=0; i<$TOTAL_PROCS; i++ ))
do
    echo "Launching process $i / $TOTAL_PROCS"
    PROC_RANK=$i TOTAL_PROCS=$TOTAL_PROCS OUTPUT_DIR=$OUTPUT_DIR python preprocess_data.py &
done

wait

echo "All preprocessing processes completed"
