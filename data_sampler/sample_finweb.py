from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from multiprocessing import freeze_support
from language_codes import language_codes
import os

"""
Donwloads data from fineweb-2 dataset based on https://huggingface.co/datasets/HuggingFaceFW/fineweb-2#using-huggingface_hub
- You can specify which languages by selecting from language_codes list in language_codes.py
- limit specifies how much data you want to download 
-> for the top-20 languages with the most data a limit of 500.00 is ~30GB data (entirely different for low-resource langauges)
"""

# pick languages
top_20_languages = language_codes[:20]

OUTPUT_DIR = "./fineweb2_subset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    TASKS = []

    for lang in top_20_languages:
        reader_path = f"hf://datasets/HuggingFaceFW/fineweb-2/data/{lang}/train"
        output_path = os.path.join(OUTPUT_DIR, f"{lang}.jsonl")

        TASKS.append(LocalPipelineExecutor(
            pipeline=[
                ParquetReader(reader_path, limit=100000),
                JsonlWriter(output_path)
            ],
            tasks=3 # amount of processes which download data concurrently
        ))

    for executor in TASKS:
        executor.run()

if __name__ == '__main__':    
    freeze_support()
    main()
