#!/bin/bash

################### Download datasets ###################
CACHE_DIR="/data/hf_cache/"

DATASETS_DOWNLOAD=(
    "angeluriot/french_instruct:"
    "DiscoResearch/germanrag:"
)
DATASETS_STRING="${DATASETS_DOWNLOAD[@]}"

echo "Start downlaoding the datasets"
python download_datasets.py --datasets $DATASETS_STRING --hf_cache_dir $CACHE_DIR
echo "Download of all datasets finished"


################### Preprocessing datasets ###################
export HF_HOME="$CACHE_DIR"
OUTPUT_DIR="/data/joel/prepared/langauge_adapters"
TOTAL_PROCS=2
TOKENIZER=xlm-roberta-base
mkdir -p $OUTPUT_DIR
#define datasets here agian like this dataset:dataset_config:tokenizer
DATASETS_PREPROCESS=(
    "angeluriot/french_instruct::$TOKENIZER"
    "DiscoResearch/germanrag::$TOKENIZER"
)
DATASETS_PREPROCESS_STRING="${DATASETS_PREPROCESS[@]}"

for (( i=0; i<$TOTAL_PROCS; i++ ))
do
    echo "Launching process $i / $TOTAL_PROCS"
    PROC_RANK=$i TOTAL_PROCS=$TOTAL_PROCS OUTPUT_DIR=$OUTPUT_DIR \
    python preprocess_data.py --output_dir $OUTPUT_DIR --datasets $DATASETS_PREPROCESS_STRING --tokenizer $TOKENIZER --log_level info &
done

wait

echo "All preprocessing processes completed"
