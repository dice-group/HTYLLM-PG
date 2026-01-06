from pathlib import Path
from safetensors.torch import load_file

ckpt = Path("/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/cola_colaexp-lpr_20260106_221852/checkpoint-40_adapter")
sd = load_file(ckpt/"adapter_model.safetensors")

print("lora keys:", sum("lora_" in k for k in sd))
print("router keys:", sum(".router." in k for k in sd))
print("expert keys:", sum(".expert_" in k for k in sd))


from torch.distributed.checkpoint import FileSystemReader
from pathlib import Path
shard = Path("/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/cola_colaexp-lpr_20260106_221852/checkpoint-40_adapter_sharded")
meta = FileSystemReader(str(shard)).read_metadata()
print("num keys:", len(meta.state_dict_metadata))
print("sample:", list(meta.state_dict_metadata.keys())[:10])