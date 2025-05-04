from multiprocessing import Pool, freeze_support
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from language_codes import language_codes, belebele_languages
import os

OUTPUT_DIR = "/data/fineweb2_subset_belebele"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_executor(lang):
    if lang not in language_codes:
        print(f"Skipping unknown language: {lang}")
        return
    reader_path = f"hf://datasets/HuggingFaceFW/fineweb-2/data/{lang}/train"
    output_path = os.path.join(OUTPUT_DIR, f"{lang}.jsonl")

    executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(reader_path, limit=20000),
            JsonlWriter(output_path)
        ],
        tasks=3
    )
    executor.run()

def main():
    max_parallel_languages = 5  # Adjust this to control parallelism
    languages_to_process = [lang for lang in belebele_languages if lang in language_codes]

    with Pool(processes=max_parallel_languages) as pool:
        pool.map(run_executor, languages_to_process)

if __name__ == '__main__':
    freeze_support()
    main()
