from datasets import Dataset

ds = Dataset.from_file("/data/project_data/moe_study/tokenized/test_llama_3.2-1B_multilingual_tok/data-00000-of-00508.arrow")
ds.save_to_disk("/data/project_data/moe_study/tokenized/test_llama_3.2-1B_multilingual_tok/")
