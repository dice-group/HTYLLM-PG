python train.py \
  --anchor_model_dir google/gemma-2b \
  --aug_model_dir NickyNicky/gemma-2b-it_oasst2_Cluster_2_aya_dataset_multilingual_chatml_response_json_V1 \
  --output_dir ./output_calm_multilingual \
  --learning_rate 3e-4 \
  --batch_size 2 \
  --epochs 3 \
  --num_heads 2 \
  --num_connections 2 \
  --max_steps 1000
