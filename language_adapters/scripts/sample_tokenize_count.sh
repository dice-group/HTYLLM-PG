#!/bin/bash

# For langauge adapter training, we first subsample data, 
# then tokenize it,
# then count amount of tokens. 
# Afterwards data is tokenized and saved and thus prepared for training
# TODO: make parts of pipeline concurrent and more efficient if needed, currently not necessary


# ----- 1. subsample from original fineweb subset ----- 
#LANGS="apc_Arab,arz_Arab,ary_Arab,ars_Arab,heb_Hebr"
LANGS="vie_Latn,tha_Thai,khm_Khmr,lao_Lao,msa_Latn" #south asian
# LANGS=heb_Hebr #test with only hebrew
# LANGS="hin_Deva,urd_Arab,mar_Deva,ben_Beng,pan_Guru" #indo-aryan lanauges 25000 samplesfor each
# LANGS="swh_Latn,yor_Latn,ibo_Latn,wol_Latn,zul_Latn" #niger-congo lanauges
# LANGS="spa_Latn"
INPUT_DIR="/data/upb/users/j/joeldag/profiles/unix/cs/fineweb2_subset"
SUB_SAMPLE_OUTPUT_DIR="/data/upb/users/j/joeldag/profiles/unix/cs/language_adapter_subsets/south_asian"
LINES=45000 # this is per langauge###

python subsample_data.py \
  --langs "$LANGS" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$SUB_SAMPLE_OUTPUT_DIR" \
  --lines "$LINES"


# ----- 2. Toknenized subsampled data ----- 
#DATA_DIR=$SUB_SAMPLE_OUTPUT_DIR
DATA_DIR="/upb/users/j/joeldag/profiles/unix/cs/language_adapter_subsets/south_asian/"
TOKENIZED_DATA_DIR="/upb/users/j/joeldag/profiles/unix/cs/tokenized_data_subsets/mistral7b/south_asian/"
MODEL_NAME="mistralai/Mistral-7B-v0.3"
CACHE_DIR="/data/upb/users/j/joeldag/profiles/unix/cs/cache_dir"
NUM_PROC=48

python tokenize_subsampled_data.py \
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
