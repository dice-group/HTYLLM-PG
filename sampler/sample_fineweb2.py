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

def load_data(total_gb: float, num_languages: int | str, dont_include_english: bool, output_dir: str, meta_file: str):
    metadata_list = load_metadata(meta_file)
    
    if num_languages == "all":
        num_languages = len(metadata_list)
    
    # Sort by disk size in DESCENDING order (largest first)
    sorted_languages = sorted(metadata_list, key=lambda x: x['Disk_size_GB'], reverse=True)
    
    selected_languages = sorted_languages[:num_languages]
    
    TASKS = []
    
    total_languages = num_languages + (0 if dont_include_english else 1)
    fair_share_gb = total_gb / total_languages
    
    remaining_gb = total_gb
    successful_languages = 0
    
    print(f"\n{'='*80}")
    print(f"Sampling {total_gb}GB from {num_languages} fineweb-2 languages" + 
          ("" if dont_include_english else " + English"))
    print(f"Initial fair share per language: {fair_share_gb:.2f}GB")
    print(f"{'='*80}\n")
    
    for i, lang in enumerate(selected_languages):
        available_gb = lang['Disk_size_GB']
        gb_to_sample = min(fair_share_gb, available_gb)
        
        if gb_to_sample > 0:
            docs_to_sample, actual_gb = calculate_documents_from_gb(lang, gb_to_sample)
            
            lang_name = lang['Subset'].strip('`')
            reader_path = f"hf://datasets/HuggingFaceFW/fineweb-2/data/{lang_name}/train"
            output_path = os.path.join(output_dir, f"{lang_name}.jsonl")
            
            print(f"Language: {lang['Name']} ({lang_name})")
            print(f"  Available: {available_gb:.2f}GB ({lang['Documents']:,} docs)")
            print(f"  Sampling: {actual_gb:.2f}GB (~{docs_to_sample:,} docs)")
            print(f"  Output: {output_path}\n")
            
            TASKS.append((lang_name, actual_gb, LocalPipelineExecutor(
                pipeline=[
                    ParquetReader(reader_path, limit=docs_to_sample),
                    JsonlWriter(output_path)
                ],
                tasks=1
            )))
            
            remaining_gb -= actual_gb
            successful_languages += 1
        
        # Recalculate fair share for remaining languages
        remaining_languages = num_languages - i - 1 + (0 if dont_include_english else 1)
        if remaining_languages > 0:
            fair_share_gb = remaining_gb / remaining_languages
    
    # Add English with remaining GB
    if not dont_include_english:
        english_gb_to_sample = remaining_gb
        
        if english_gb_to_sample > 0:
            # Estimate English documents needed
            # Rough estimate: ~1.6MB per document compressed
            estimated_docs_per_gb = 640  # ~1.6MB per doc
            english_docs_to_sample = int(english_gb_to_sample * estimated_docs_per_gb)
            
            english_output_path = os.path.join(output_dir, "english.jsonl")
            
            print(f"Language: English (eng_Latn)")
            print(f"  Sampling: {english_gb_to_sample:.2f}GB (~{english_docs_to_sample:,} docs)")
            print(f"  Output: {english_output_path}\n")
            
            english_pipeline = LocalPipelineExecutor(
                pipeline=[
                    ParquetReader("hf://datasets/HuggingFaceFW/fineweb/data/CC-MAIN-2024-10", 
                                 limit=english_docs_to_sample),
                    JsonlWriter(english_output_path)
                ],
                tasks=1
            )
            TASKS.append(("english", english_gb_to_sample, english_pipeline))
    
    print(f"{'='*80}")
    print(f"Total languages to process: {len(TASKS)}")
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