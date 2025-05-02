from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from multiprocessing import freeze_support
import os
import logging
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

"""
Samples data from fineweb-2 dataset based on a specified strategy.
This script accepts command line arguments to configure the sampling process:

Usage:
    python sampler/sample_fineweb2.py --total_docs 10000 --num_languages 500 --output_dir ./fineweb2_subset

Parameters:
    --total_docs: Total number of documents to sample across all languages
    --num_languages: Number of languages to include in the sampling
    --output_dir: Directory to store the sampled data
    --stats_file: File to store sampling statistics (default: sampling_stats.json in output_dir)
    --meta_file: Path to the fineweb2 metadata file containing document counts
    --include_english: Include English data from fineweb dataset with remaining budget
    --log_level: Logging level (default: INFO)
"""

def setup_logging(log_level):
    """Configure logging with the specified level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(level=numeric_level, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Sample data from fineweb-2 dataset.')
    parser.add_argument('--total_docs', type=int, default=10_000, help='Total number of documents to sample across all languages')
    parser.add_argument('--num_languages', type=int, default=500, help='Number of languages to include in the sampling')
    parser.add_argument('--output_dir', type=str, default='./fineweb2_subset', help='Directory to store the sampled data')
    parser.add_argument('--stats_file', type=str, default=None, help='File to store sampling statistics (default: sampling_stats.json in output_dir)')
    parser.add_argument('--meta_file', type=str, default='./sampler/fineweb2_meta.json', help='Path to the fineweb2 metadata file containing document counts')
    parser.add_argument('--include_english', action='store_true', help='Include English data from fineweb dataset with remaining budget')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    
    return parser.parse_args()

def load_metadata(meta_file):
    with open(meta_file, 'r') as f:
        return json.load(f)

def main():
    args = parse_arguments()
    logger = setup_logging(args.log_level)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    stats_file = args.stats_file or os.path.join(output_dir, "sampling_stats.json")
    
    logger.info(f"Loading metadata from: {args.meta_file}")
    metadata_list = load_metadata(args.meta_file)
    logger.info(f"Loaded metadata for {len(metadata_list)} languages")
    
    # Ensure metadata is sorted by document count in descending order
    metadata_list = sorted(metadata_list, key=lambda x: x['Documents'], reverse=True)
    
    # Extract language subset identifiers and create a mapping for quick lookup
    language_meta_map = {}
    sorted_languages = []
    
    for item in metadata_list:
        # Extract the subset identifier, removing backticks if present
        subset = item['Subset'].replace('`', '')
        language_meta_map[subset] = item
        sorted_languages.append(subset)
    
    logger.info(f"Found {len(sorted_languages)} languages in metadata")
    
    # Adjust for English if include_english flag is set
    num_languages_to_select = args.num_languages
    if args.include_english:
        num_languages_to_select -= 1  # Reserve one spot for English
        logger.info(f"Including English, reducing other languages to {num_languages_to_select}")
    
    # Select languages (take the first N most resourced languages)
    selected_languages = sorted_languages[:min(num_languages_to_select, len(sorted_languages))]
    logger.info(f"Selected {len(selected_languages)} languages from most resourced")
    
    # Reverse the order for processing to start from least resourced
    processing_order = selected_languages.copy()
    processing_order.reverse()
    logger.info(f"Processing languages from least to most resourced")
    
    remaining_docs = args.total_docs
    remaining_langs = len(processing_order)
    
    stats = {
        "parameters": {
            "total_target": args.total_docs,
            "num_languages": args.num_languages,
            "include_english": args.include_english,
            "output_dir": output_dir,
            "metadata_file": args.meta_file
        },
        "total_actual": 0,
        "languages": {}
    }
    
    # Create tasks list for execution
    TASKS = []
    
    for lang_subset in processing_order:
        if remaining_docs <= 0 or remaining_langs <= 0:
            break
            
        fair_share = remaining_docs // remaining_langs
        
        # Get actual document count from metadata
        lang_meta = language_meta_map[lang_subset]
        available_docs = lang_meta['Documents']
        
        docs_to_sample = min(fair_share, available_docs)
        
        logger.info(f"Language: {lang_subset} ({lang_meta['Name']}), Fair share: {fair_share}, Available: {available_docs}, Sampling: {docs_to_sample}")
        
        if docs_to_sample > 0:
            reader_path = f"hf://datasets/HuggingFaceFW/fineweb-2/data/{lang_subset}/train"
            output_path = os.path.join(output_dir, f"{lang_subset}.jsonl")
            
            TASKS.append(LocalPipelineExecutor(
                pipeline=[
                    ParquetReader(reader_path, limit=docs_to_sample),
                    JsonlWriter(output_path)
                ],
                tasks=1  # Reduced from 4 to 1 to avoid rate limiting
            ))
            
            # Update tracking variables
            remaining_docs -= docs_to_sample
            remaining_langs -= 1
            
            # Record stats
            stats["languages"][lang_subset] = {
                "name": lang_meta['Name'],
                "fair_share": fair_share,
                "available": available_docs,
                "sampled": docs_to_sample
            }
            stats["total_actual"] += docs_to_sample
        else:
            remaining_langs -= 1
            stats["languages"][lang_subset] = {
                "name": lang_meta['Name'],
                "fair_share": fair_share,
                "available": available_docs,
                "sampled": 0,
                "reason": "No documents to sample"
            }
        
        # Stop if we've exhausted our document budget
        if remaining_docs <= 0:
            logger.info(f"Document budget exhausted after {lang_subset}")
            break
    
    # Handle English separately if flag is set and we have remaining docs
    if args.include_english and remaining_docs > 0:
        logger.info(f"Using remaining document budget ({remaining_docs}) for English data")
        
        english_output_path = os.path.join(output_dir, "english.jsonl")
        
        # Add English data pipeline
        english_pipeline = LocalPipelineExecutor(
            pipeline=[
                ParquetReader("hf://datasets/HuggingFaceFW/fineweb/data/CC-MAIN-2024-10", limit=remaining_docs),
                JsonlWriter(english_output_path)
            ],
            tasks=1  # Using 1 task to avoid rate limiting
        )
        
        try:
            logger.info(f"Processing English data, sampling up to {remaining_docs} documents")
            english_pipeline.run()
            stats["languages"]["english"] = {
                "name": "English",
                "fair_share": remaining_docs,
                "available": "unknown",  # We don't know the total available count
                "sampled": remaining_docs
            }
            stats["total_actual"] += remaining_docs
            logger.info(f"Completed English data sampling")
        except Exception as exc:
            logger.error(f"English data sampling generated an exception: {exc}")
            stats["languages"]["english"] = {
                "name": "English",
                "error": str(exc),
                "sampled": 0
            }

    logger.info(f"Executing {len(TASKS)} sampling tasks sequentially")

    for i, executor in enumerate(TASKS):
        task_idx = i + 1
        logger.info(f"Running task {task_idx}/{len(TASKS)}")
        try:
            executor.run()
            logger.info(f"Completed task {task_idx}/{len(TASKS)}")
        except Exception as exc:
            logger.error(f"Task {task_idx} generated an exception: {exc}")

    # Save sampling statistics
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Sampling completed. Total documents sampled: {stats['total_actual']}")
    logger.info(f"Statistics saved to {stats_file}")

if __name__ == '__main__':
    freeze_support()
    main()