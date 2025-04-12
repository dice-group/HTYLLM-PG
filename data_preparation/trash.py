import os
from datasets import load_dataset

os.environ["HF_HOME"] = "/data/hf_cache/"

dataset = load_dataset("allenai/c4", "realnewslike", split="train")
