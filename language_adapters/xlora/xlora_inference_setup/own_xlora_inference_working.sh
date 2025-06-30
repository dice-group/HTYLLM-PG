RUST_BACKTRACE=full ./target/debug/mistralrs-server \
  --port 1234 \
  x-lora \
  --xlora-model-id /data/joel/results_language_adapters/xlora/mistral7b/depth_1_jav_Latn_sun_Latn_swh_Latn_sna_Latn_nya_Latn/checkpoint-50/ \
  -o /data/joel/results_language_adapters/xlora/mistral7b/depth_1_jav_Latn_sun_Latn_swh_Latn_sna_Latn_nya_Latn/ordering.json 
