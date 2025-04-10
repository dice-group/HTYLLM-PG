#!/bin/bash

export TOTAL_PROCS=32
export OUTPUT_DIR="/data/output_realnews"

mkdir -p $OUTPUT_DIR

#spawn processes
for ((i=0; i<$TOTAL_PROCS; i++))
do
    echo "Launching training process $i / $TOTAL_PROCS"
    PROC_RANK=$i TOTAL_PROCS=$TOTAL_PROCS OUTPUT_DIR=$OUTPUT_DIR python3 train_test.py &
done

wait
echo "Training finishe"
