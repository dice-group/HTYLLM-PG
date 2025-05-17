#!/bin/bash
"""
For langauge adapter training, we first subsample data, 
then tokenize it,
then count amount of tokens. 
Afterwards data is tokenized and saved and thus prepared for training
TODO: make parts of pipeline concurrent and more efficient if needed, currently not necessary
"""

# ----- 1. subsample from original fineweb subset ----- 
LANGS="apc_Arab,arz_Arab,ary_Arab,ars_Arab,heb_Hebr"
INPUT_DIR="/data/fineweb2_subset"
SUB_SAMPLE_OUTPUT_DIR="/data/joel/language_adapters_subsets/arabic_subset"
LINES=28000 # this is per langauge

python subsample.py \
  --langs "$LANGS" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$SUB_SAMPLE_OUTPUT_DIR" \
  --lines "$LINES"


# ----- 2. Toknenized subsampled data ----- 
DATA_DIR=$SUB_SAMPLE_OUTPUT_DIR
TOKENIZED_DATA_DIR="/data/joel/tokenized_adapter_subsets/arabic_subset/"
MODEL_NAME="bigscience/bloom-560m"
CACHE_DIR="/data/shared_home_cache/datasets/joel"
NUM_PROC=24

python script_name.py \
  --data_dir "$DATA_DIR" \
  --tokenized_data_dir "$TOKENIZED_DATA_DIR" \
  --model_name "$MODEL_NAME" \
  --cache_dir "$CACHE_DIR" \
  --num_proc "$NUM_PROC"



#----- 3. check how many tokens there are in data ----- 
DATASET_PATH=$TOKENIZED_DATA_DIR
NUM_PROC=24

python count_tokens.py \
  --dataset_path "$DATASET_PATH" \
  --num_proc "$NUM_PROC"
