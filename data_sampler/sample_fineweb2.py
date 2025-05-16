from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from multiprocessing import freeze_support
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
import logging
import json
import argparse
import time
import random
import requests
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description='Sample data from fineweb-2 dataset.')
    parser.add_argument('--total_docs', type=int, default=10_000)
    parser.add_argument('--num_languages', type=lambda x: int(x) if x != 'all' else 'all', default=500,
                        help='Number of languages to sample from, or "all" to use all available languages')
    parser.add_argument('--output_dir', type=str, default='./fineweb2_subset')
    parser.add_argument('--meta_file', type=str, default='./sampler/fineweb2_meta.json')
    parser.add_argument('--dont_include_english', action='store_true')
    return parser.parse_args()


def load_metadata(meta_file):
    with open(meta_file, 'r') as f:
        return json.load(f)


def load_data(total_docs: int, num_languages: int | str, dont_include_english: bool, output_dir: str, meta_file: str):
    metadata_list = load_metadata(meta_file)
    if num_languages == "all":
        num_languages = len(metadata_list)
        
    results = {}

    # Sort languages by number of documents in descending order
    sorted_languages = sorted(metadata_list, key=lambda x: x['Documents'], reverse=True)

    # Select the top num_languages languages
    selected_languages = sorted_languages[:num_languages]
    # reverse selected languages to sample from least to most documents
    selected_languages = selected_languages[::-1]
    TASKS = []
    if not dont_include_english:
        fair_share_per_lang = total_docs // num_languages
    else:
        fair_share_per_lang = total_docs // (num_languages + 1)

    remaining_docs = total_docs
    for i, lang in enumerate(selected_languages):
        print(f"Current fair share per language: {fair_share_per_lang} of language {lang['Subset']} which has {lang['Documents']} documents")
        available_docs = lang['Documents']
        docs_to_sample = min(fair_share_per_lang, available_docs)
        results[lang['Subset']] = docs_to_sample
        print(f"Sampling {docs_to_sample} documents for language {lang['Subset']}")

        if docs_to_sample > 0:
            lang_name = lang['Subset'].strip('`')
            reader_path = f"hf://datasets/HuggingFaceFW/fineweb-2/data/{lang_name}/train"
            output_path = os.path.join(output_dir, f"{lang_name}.jsonl")

            TASKS.append((lang, LocalPipelineExecutor(
                pipeline=[
                    ParquetReader(reader_path, limit=docs_to_sample),
                    JsonlWriter(output_path)
                ],
                tasks=1
            )))

        remaining_docs -= docs_to_sample
        if dont_include_english:
            fair_share_per_lang = remaining_docs // (num_languages - i)
        else:
            fair_share_per_lang = remaining_docs // (num_languages - i + 1)

    if not dont_include_english:
        english_docs_to_sample = max(fair_share_per_lang, remaining_docs)
        results["english"] = english_docs_to_sample

        if english_docs_to_sample > 0:
            english_output_path = os.path.join(output_dir, "english.jsonl")

            english_pipeline = LocalPipelineExecutor(
                pipeline=[
                    ParquetReader("hf://datasets/HuggingFaceFW/fineweb/data/CC-MAIN-2024-10", limit=english_docs_to_sample),
                    JsonlWriter(english_output_path)
                ],
                tasks=1
            )
            TASKS.append(("english", english_pipeline))
            
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(task.run): lang['Subset'] if isinstance(lang, dict) else lang for lang, task in TASKS}
        for future in as_completed(futures):
            lang = futures[future]
            try:
                future.result()
                print(f"{lang} completed.")
            except Exception as e:
                print(f"Error processing {lang}: {e}")

        
    with open(os.path.join(output_dir, "sampling_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

def main():
    args = parse_arguments()
    load_data(args.total_docs, args.num_languages, args.dont_include_english, args.output_dir, args.meta_file)

if __name__ == '__main__':
    freeze_support()
    main()