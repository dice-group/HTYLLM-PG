#!/bin/bash

################### Download datasets ###################
CACHE_DIR="/data/hf_cache/"

DATASETS_DOWNLOAD=(
    "wikitext:wikitext-2-raw-v1"
    "ag_news:"
)
DATASETS_STRING="${DATASETS_DOWNLOAD[@]}"

echo "Start downlaoding the dataset"
python download_datasets.py --datasets $DATASETS_STRING --hf_cache_dir $CACHE_DIR
echo "Download of all datasets finished"


################### Preprocessing datasets ###################
export HF_HOME="$CACHE_DIR"
OUTPUT_DIR="/data/joel/prepared"
TOTAL_PROCS=32
mkdir -p $OUTPUT_DIR
#define datasets here agian like this dataset:dataset_config:tokenizer
DATASETS_PREPROCESS=(
    "wikitext:wikitext-2-raw-v1:gpt2"
    "ag_news:default:gpt2"
)
DATASETS_PREPROCESS_STRING="${DATASETS_PREPROCESS[@]}"

for (( i=0; i<$TOTAL_PROCS; i++ ))
do
    echo "Launching process $i / $TOTAL_PROCS"
    PROC_RANK=$i TOTAL_PROCS=$TOTAL_PROCS OUTPUT_DIR=$OUTPUT_DIR \
    python preprocess_data.py --output_dir $OUTPUT_DIR --datasets $DATASETS_PREPROCESS_STRING &
done

wait

echo "All preprocessing processes completed"
