RUST_BACKTRACE=1 ./mistralrs-server \
  --port 1234 \
  x-lora \
  --xlora-model-id /data/joel/results_language_adapters/xlora/mistral7b/swh_Latn_sna_Latn_nya_Latn_south_asian/checkpoint-2000/ \
  -o /data/joel/results_language_adapters/xlora/mistral7b/swh_Latn_sna_Latn_nya_Latn_south_asian/my_ordering.json
