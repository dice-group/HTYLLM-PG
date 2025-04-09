#!/bin/bash

export TOTAL_PROCS=32
export OUTPUT_DIR="/data/output_realnews"
mkdir -p $OUTPUT_DIR

#spawn processes
for (( i=0; i<$TOTAL_PROCS; i++ ))
do
    echo "Launching process $i / $TOTAL_PROCS"
    PROC_RANK=$i TOTAL_PROCS=$TOTAL_PROCS OUTPUT_DIR=$OUTPUT_DIR python preprocess_data.py &
done

wait

echo "All preprocessing processes completed"
