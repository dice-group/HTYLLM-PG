from pathlib import Path
from safetensors.torch import load_file, save_file

ckpt = Path("/data/joel/results_language_adapters/xlora/mistral7b/depth_1_jav_Latn_sun_Latn_swh_Latn_sna_Latn_nya_Latn/checkpoint-50")

for sub in ["adapter_1", "adapter_2"]:
    f = ckpt / sub / "adapter_model.safetensors"
    sd = load_file(f)
    keys_to_drop = [k for k in sd if k.startswith("internal_xlora_classifier.")]
    if keys_to_drop:
        print(f"{f}  →  removing {len(keys_to_drop)} classifier tensors")
        for k in keys_to_drop:
            del sd[k]
        save_file(sd, f)
        print("saved cleaned file")
