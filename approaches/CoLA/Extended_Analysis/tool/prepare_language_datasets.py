#!/usr/bin/env python3
"""
Prepare language-specific test datasets for expert routing analysis.

This script works with FineWeb2-style datasets where data is organized 
in subdirectories by language (e.g., samples/en_Latn/, samples/es_Latn/).

Usage:
    python prepare_language_datasets.py \
        --validation_data /path/to/samples \
        --languages "en,es,fr,de,zh" \
        --num_sequences 10000 \
        --output_dir ./data/language_test_sets
"""

import argparse
import json
import gzip
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Language code mapping (actual directory names to simple codes)
# Based on ISO 639-3 codes used in FineWeb2 samples
LANG_CODE_MAPPING = {
    # Special case
    'english': 'en',
    
    # Major languages (ISO 639-3 -> ISO 639-1)
    'spa_Latn': 'es',   # Spanish
    'rus_Cyrl': 'ru',   # Russian
    'hin_Deva': 'hi',   # Hindi (Devanagari)
    'hin_Latn': 'hi',   # Hindi (Latin)
    'asm_Beng': 'as',   # Assamese
    'ben_Latn': 'bn',   # Bengali (Latin)
    'ory_Orya': 'or',   # Oriya
    'tel_Telu': 'te',   # Telugu
    'tel_Latn': 'te',   # Telugu (Latin)
    'tcy_Knda': 'kn',   # Kannada variant
    'kat_Geor': 'ka',   # Georgian
    'fas_Arab': 'fa',   # Farsi/Persian
    'azj_Cyrl': 'az',   # Azerbaijani (Cyrillic)
    'tat_Latn': 'tt',   # Tatar
    'cym_Latn': 'cy',   # Welsh
    'slv_Latn': 'sl',   # Slovenian
    'fin_Latn': 'fi',   # Finnish
    'dzo_Tibt': 'dz',   # Dzongkha
    'lao_Laoo': 'lo',   # Lao
    'csy_Latn': 'cs',   # Czech variant
    'srp_Cyrl': 'sr',   # Serbian (Cyrillic)
    'sma_Latn': 'se',   # Sami
    'arz_Arab': 'ar',   # Egyptian Arabic
    'sdh_Arab': 'ku',   # Kurdish (Sorani)
    'hbo_Hebr': 'he',   # Ancient Hebrew
    'nya_Latn': 'ny',   # Chichewa
    'sbd_Latn': 'mg',   # Southern dialect (related to Malagasy)
    'snd_Deva': 'sd',   # Sindhi
    'uzs_Arab': 'uz',   # Uzbek (Arabic script)
    'sat_Olck': 'sat',  # Santali
    'mni_Beng': 'mni',  # Manipuri
    'dty_Deva': 'ne',   # Nepali variant
    'nwx_Deva': 'ne',   # Another Nepali variant
    'kru_Deva': 'ks',   # Kashmiri
    'sah_Cyrl': 'sah',  # Sakha/Yakut
    'mhr_Cyrl': 'mhr',  # Meadow Mari
    'myb_Latn': 'my',   # Burmese variant
    'ksw_Mymr': 'ksw',  # S'gaw Karen
    'kxm_Thai': 'kxm',  # Northern Khmer
    'wuu_Hani': 'zh',   # Wu Chinese
    'zsm_Latn': 'ms',   # Malay
    'jav_Latn': 'jv',   # Javanese
    'bug_Latn': 'bug',  # Buginese
    'tso_Latn': 'ts',   # Tsonga
    'jbo_Latn': 'jbo',  # Lojban
    'ile_Latn': 'ie',   # Interlingue
    'tig_Ethi': 'ti',   # Tigrinya
    'syc_Syrc': 'syc',  # Classical Syriac
    'ang_Latn': 'ang',  # Old English
    'orv_Cyrl': 'orv',  # Old Russian
    'prg_Latn': 'prg',  # Old Prussian
    'xal_Cyrl': 'xal',  # Kalmyk
    'koi_Cyrl': 'koi',  # Komi-Permyak
    'tkr_Cyrl': 'tkr',  # Tsakhur
    'abk_Cyrl': 'ab',   # Abkhaz
    'tzm_Tfng': 'tzm',  # Tamazight
    
    # Add more as needed based on your actual data
}



def find_language_directories(base_path: Path, target_languages: List[str]) -> Dict[str, Path]:
    """
    Find subdirectories for target languages in FineWeb2 structure.
    
    Args:
        base_path: Base samples directory
        target_languages: List of simple language codes (e.g., ['en', 'es'])
    
    Returns:
        Dictionary mapping simple lang code to directory path
    """
    logger.info(f"Scanning {base_path} for language directories...")
    
    lang_to_dir = {}
    
    # Create reverse mapping (simple code -> full code with script)
    simple_to_full = defaultdict(list)
    for full_code, simple_code in LANG_CODE_MAPPING.items():
        simple_to_full[simple_code].append(full_code)
    
    # Scan for directories
    if not base_path.exists():
        raise FileNotFoundError(f"Base path does not exist: {base_path}")
    
    for lang_dir in base_path.iterdir():
        if not lang_dir.is_dir():
            continue
        
        dir_name = lang_dir.name
        
        # Check if this directory corresponds to a target language
        simple_code = LANG_CODE_MAPPING.get(dir_name)
        
        if simple_code in target_languages:
            # Prefer first matching directory for each language
            if simple_code not in lang_to_dir:
                lang_to_dir[simple_code] = lang_dir
                logger.info(f"Found '{simple_code}' data in {dir_name}/")
    
    # Check which languages were not found
    missing = set(target_languages) - set(lang_to_dir.keys())
    if missing:
        logger.warning(f"No data found for languages: {missing}")
    
    return lang_to_dir


def read_jsonl_files(directory: Path, num_sequences: int, text_field: str = 'text') -> List[Dict[str, Any]]:
    """
    Read sequences from compressed or uncompressed JSONL files in a directory.
    
    Args:
        directory: Directory containing JSONL files
        num_sequences: Number of sequences to extract
        text_field: Field name containing text (default: 'text')
    
    Returns:
        List of examples
    """
    examples = []
    
    # Find all JSONL files (compressed or not)
    jsonl_files = sorted(list(directory.glob('*.jsonl.gz')) + list(directory.glob('*.jsonl')))
    
    if not jsonl_files:
        logger.warning(f"No JSONL files found in {directory}")
        return examples
    
    logger.info(f"Found {len(jsonl_files)} JSONL files in {directory.name}/")
    
    for jsonl_file in jsonl_files:
        if len(examples) >= num_sequences:
            break
        
        # Open file (handle both compressed and uncompressed)
        if jsonl_file.suffix == '.gz':
            f = gzip.open(jsonl_file, 'rt', encoding='utf-8')
        else:
            f = open(jsonl_file, 'r', encoding='utf-8')
        
        try:
            for line in f:
                if len(examples) >= num_sequences:
                    break
                
                try:
                    example = json.loads(line.strip())
                    
                    # Verify text field exists
                    if text_field in example and example[text_field]:
                        examples.append(example)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line in {jsonl_file.name}: {e}")
                    continue
        
        finally:
            f.close()
    
    return examples


def extract_language_sequences(
    base_path: Path,
    languages: List[str],
    num_sequences: int,
    text_field: str = 'text'
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract sequences per language from FineWeb2 directory structure.
    
    Args:
        base_path: Base directory containing language subdirectories
        languages: List of language codes to extract (simple codes like 'en', 'es')
        num_sequences: Number of sequences per language
        text_field: Field name containing text
    
    Returns:
        Dictionary mapping language code to list of examples
    """
    logger.info(f"Extracting {num_sequences} sequences for {len(languages)} languages")
    
    # Find language directories
    lang_to_dir = find_language_directories(Path(base_path), languages)
    
    if not lang_to_dir:
        raise ValueError("No language directories found!")
    
    # Extract sequences from each language directory
    language_data = {}
    
    for lang, lang_dir in tqdm(lang_to_dir.items(), desc="Processing languages"):
        logger.info(f"Reading data for '{lang}' from {lang_dir.name}/")
        
        examples = read_jsonl_files(lang_dir, num_sequences, text_field)
        
        if examples:
            language_data[lang] = examples
            logger.info(f"✓ Collected {len(examples)} sequences for '{lang}'")
        else:
            logger.warning(f"✗ No valid sequences found for '{lang}'")
    
    return language_data


def save_language_datasets(
    language_data: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    text_field: str = 'text'
):
    """Save language-specific datasets to JSONL files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for lang, examples in language_data.items():
        output_file = output_dir / f"{lang}.jsonl"
        
        logger.info(f"Saving {len(examples)} examples for '{lang}' to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                # Keep only essential fields to save space
                cleaned_example = {
                    'text': example.get(text_field, ''),
                    'id': example.get('id', ''),
                    'language': lang  # Add explicit language field
                }
                f.write(json.dumps(cleaned_example, ensure_ascii=False) + '\n')
    
    # Create metadata file
    metadata_file = output_dir / "metadata.json"
    metadata = {
        'dataset_type': 'fineweb2',
        'num_languages': len(language_data),
        'languages': list(language_data.keys()),
        'sequences_per_language': {
            lang: len(examples) for lang, examples in language_data.items()
        },
        'total_sequences': sum(len(examples) for examples in language_data.values())
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata to {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare language-specific test datasets for expert routing analysis"
    )
    parser.add_argument(
        '--validation_data',
        type=str,
        required=True,
        help='Path to FineWeb2 samples directory containing language subdirectories'
    )
    parser.add_argument(
        '--languages',
        type=str,
        required=True,
        help='Comma-separated list of language codes (e.g., "en,es,fr,de,zh")'
    )
    parser.add_argument(
        '--num_sequences',
        type=int,
        default=10000,
        help='Number of sequences to extract per language (default: 10000)'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        required=True,
        help='Output directory for language-specific datasets'
    )
    parser.add_argument(
        '--text_field',
        type=str,
        default='text',
        help='Field name containing text content (default: "text")'
    )
    
    args = parser.parse_args()
    
    # Parse languages
    languages = [lang.strip() for lang in args.languages.split(',')]
    logger.info(f"Target languages: {languages}")
    
    # Extract language-specific sequences
    language_data = extract_language_sequences(
        Path(args.validation_data),
        languages,
        args.num_sequences,
        args.text_field
    )
    
    if not language_data:
        logger.error("No data extracted! Check language codes and directory structure.")
        return
    
    # Save to disk
    save_language_datasets(language_data, args.output_dir, args.text_field)
    
    logger.info("Dataset preparation complete!")


if __name__ == '__main__':
    main()

