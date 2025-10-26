from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from multiprocessing import freeze_support
import os
import logging
import json
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Sample data from fineweb-2 dataset.')
    parser.add_argument('--total_gb', type=float, default=100.0,
                        help='Total gigabytes to sample')
    parser.add_argument('--num_languages', type=lambda x: int(x) if x != 'all' else 'all', default=500,
                        help='Number of languages to sample from, or "all" to use all available languages')
    parser.add_argument('--output_dir', type=str, default='./fineweb2_subset')
    parser.add_argument('--meta_file', type=str, default='./sampler/fineweb2_meta.json')
    parser.add_argument('--dont_include_english', action='store_true')
    return parser.parse_args()

def parse_disk_size(size_str):
    """Convert disk size string (e.g., '1.65TB', '640.76GB', '925.90KB') to GB."""
    size_str = size_str.strip()
    if 'TB' in size_str:
        return float(size_str.replace('TB', '')) * 1024
    elif 'GB' in size_str:
        return float(size_str.replace('GB', ''))
    elif 'MB' in size_str:
        return float(size_str.replace('MB', '')) / 1024
    elif 'KB' in size_str:
        return float(size_str.replace('KB', '')) / (1024 * 1024)
    else:
        raise ValueError(f"Unknown size format: {size_str}")

def load_metadata(meta_file):
    with open(meta_file, 'r') as f:
        metadata_list = json.load(f)
    
    # Parse disk sizes to GB
    for lang in metadata_list:
        lang['Disk_size_GB'] = parse_disk_size(lang['Disk size'])
    
    return metadata_list

def calculate_documents_from_gb(lang, target_gb):
    """Calculate approximate number of documents needed for target GB."""
    lang_total_gb = lang['Disk_size_GB']
    lang_total_docs = lang['Documents']
    
    # Proportion of data we want
    proportion = min(1.0, target_gb / lang_total_gb)
    
    # Calculate documents needed
    docs_needed = int(lang_total_docs * proportion)
    
    return docs_needed, min(target_gb, lang_total_gb)

def calculate_fair_shares(languages, total_gb):
    """
    Iteratively calculate fair shares for languages, redistributing excess
    from languages that can't provide their full share.
    
    Returns a dict mapping language index to allocated GB.
    """
    num_langs = len(languages)
    allocated = {}  # language index -> allocated GB
    remaining_langs = set(range(num_langs))
    remaining_gb = total_gb
    
    iteration = 0
    while remaining_langs and remaining_gb > 0.001:  # Stop when < 1MB left
        iteration += 1
        fair_share = remaining_gb / len(remaining_langs)
        
        # Find languages that can't provide their fair share
        capped_langs = []
        uncapped_langs = []
        
        for idx in remaining_langs:
            available = languages[idx]['Disk_size_GB']
            if available < fair_share:
                capped_langs.append(idx)
            else:
                uncapped_langs.append(idx)
        
        # Allocate to capped languages (they get all they can provide)
        for idx in capped_langs:
            allocated[idx] = languages[idx]['Disk_size_GB']
            remaining_gb -= allocated[idx]
            remaining_langs.remove(idx)
        
        if not capped_langs and remaining_langs:
            for idx in remaining_langs:
                allocated[idx] = fair_share
            remaining_gb = 0
            break
    
    return allocated

def load_data(total_gb: float, num_languages: int | str, dont_include_english: bool, output_dir: str, meta_file: str):
    metadata_list = load_metadata(meta_file)
    
    if num_languages == "all":
        num_languages = len(metadata_list)
    
    # Sort by disk size in DESCENDING order (largest first)
    sorted_languages = sorted(metadata_list, key=lambda x: x['Disk_size_GB'], reverse=True)
    
    selected_languages = sorted_languages[:num_languages]
    
    languages_to_process = selected_languages.copy()
    if not dont_include_english:
        english_entry = {
            'Name': 'English',
            'Subset': 'eng_Latn',
            'Disk_size_GB': 10000.0,  # Very large, effectively unlimited
            'Documents': 10000000  # Placeholder
        }
        languages_to_process.append(english_entry)
    
    print(f"\n{'='*80}")
    print(f"Calculating optimal distribution for {total_gb}GB across {len(languages_to_process)} languages")
    print(f"{'='*80}\n")
    
    allocations = calculate_fair_shares(languages_to_process, total_gb)
    
    TASKS = []
    print(f"Language allocations:")
    print(f"{'-'*80}\n")
    
    for idx, lang in enumerate(languages_to_process):
        gb_to_sample = allocations.get(idx, 0)
        
        if gb_to_sample > 0.001:  # Skip if less than 1MB
            lang_name = lang['Subset'].strip('`')
            
            # Special handling for English
            if lang_name == 'eng_Latn' and not dont_include_english:
                estimated_docs_per_gb = 640
                docs_to_sample = int(gb_to_sample * estimated_docs_per_gb)
                reader_path = "hf://datasets/HuggingFaceFW/fineweb/data/CC-MAIN-2024-10"
                output_path = os.path.join(output_dir, "english.jsonl")
                available_gb = "unlimited"
            else:
                docs_to_sample, actual_gb = calculate_documents_from_gb(lang, gb_to_sample)
                reader_path = f"hf://datasets/HuggingFaceFW/fineweb-2/data/{lang_name}/train"
                output_path = os.path.join(output_dir, f"{lang_name}.jsonl")
                available_gb = f"{lang['Disk_size_GB']:.2f}GB"
            
            print(f"Language: {lang['Name']} ({lang_name})")
            print(f"  Available: {available_gb}")
            print(f"  Allocated: {gb_to_sample:.2f}GB (~{docs_to_sample:,} docs)")
            print(f"  Output: {output_path}\n")
            
            TASKS.append((lang_name, gb_to_sample, LocalPipelineExecutor(
                pipeline=[
                    ParquetReader(reader_path, limit=docs_to_sample),
                    JsonlWriter(output_path)
                ],
                tasks=1
            )))
    
    print(f"{'='*80}")
    print(f"Total languages to process: {len(TASKS)}")
    print(f"Total allocated: {sum(allocations.values()):.2f}GB")
    print(f"Starting data extraction...")
    print(f"{'='*80}\n")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Execute all pipelines with error handling
    successful_downloads = []
    failed_downloads = []
    
    for lang_name, target_gb, executor in TASKS:
        print(f"Processing {lang_name}...")
        try:
            executor.run()
            print(f"Completed {lang_name}\n")
            successful_downloads.append((lang_name, target_gb))
        except Exception as e:
            print(f"Failed {lang_name}: {str(e)}\n")
            failed_downloads.append((lang_name, target_gb, str(e)))
            # Continue with next language instead of crashing
            continue
    
    # Summary
    print(f"{'='*80}")
    print(f"Data extraction complete!")
    print(f"{'='*80}")
    print(f"Successful: {len(successful_downloads)}/{len(TASKS)} languages")
    
    if successful_downloads:
        total_successful_gb = sum(gb for _, gb in successful_downloads)
        print(f"Total data downloaded: ~{total_successful_gb:.2f}GB")
        print(f"\nSuccessful languages:")
        for lang_name, gb in successful_downloads:
            print(f"  ✓ {lang_name}: ~{gb:.2f}GB")
    
    if failed_downloads:
        print(f"\nFailed languages ({len(failed_downloads)}):")
        for lang_name, gb, error in failed_downloads:
            print(f"  ✗ {lang_name}: {error[:100]}")
    
    print(f"\nOutput directory: {output_dir}")
    print(f"{'='*80}")

def main():
    args = parse_arguments()
    load_data(args.total_gb, args.num_languages, args.dont_include_english, args.output_dir, args.meta_file)

if __name__ == '__main__':
    freeze_support()
    main()