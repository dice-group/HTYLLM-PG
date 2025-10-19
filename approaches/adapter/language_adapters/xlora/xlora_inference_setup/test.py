from safetensors.torch import safe_open

classifier_path = "/data/joel/results_language_adapters/xlora/mistral7b/jav_Latn_sun_Latn_swh_Latn_sna_Latn_nya_Latn/checkpoint-1000/xlora_classifier.safetensors"

with safe_open(classifier_path, framework="pt") as f:
    keys = list(f.keys())

inner_keys = [k for k in keys if k.startswith("inner.")]
print(f"Found {len(inner_keys)} classifier layer keys:")
for k in sorted(inner_keys):
    print("  ", k)

#
#cargo build --features "cuda flash-attn"