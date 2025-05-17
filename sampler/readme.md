# Fineweb2 Sampler Guide

## Basic Usage

```bash
python sampler/sample_fineweb2.py --total_docs 10000 --num_languages 500 --output_dir ./fineweb2_subset
```

## Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--total_docs` | Documents to sample across languages | 10,000 |
| `--num_languages` | Languages to include | 500 |
| `--output_dir` | Output directory | `./fineweb2_subset` |
| `--stats_file` | Statistics file | `sampling_stats.json` |
| `--meta_file` | Metadata file | `./sampler/filtered_fineweb2_meta.json` |
| `--num_proc` | Max amount of processes used to download data concurrently | 1 |
| `--include_english` | Add English data with remaining budget | False |
| `--log_level` | Log level | INFO |

## Sampling Strategy

- Prioritizes low-resource languages first
- For each language: `fair_share = remaining_docs // remaining_langs`
- If a language has fewer documents than its fair share, it only takes what's available
- Unused budget from low-resource languages is automatically redistributed to remaining languages
- Higher-resource languages processed later benefit from this redistribution
- When `--include_english` is used, English is processed last and benefits from all previous redistributions
- `--num_proc`can be carefully used to download multiple languages concurrently to accelerate the process. Watch out to not run into HTTP 429 (to many request) errors to hugginfacce when setting this in the cluster. Use huggingface-login to increase the request limit

## Examples

```bash
# 5,000 docs from 100 languages
python sampler/sample_fineweb2.py --total_docs 5000 --num_languages 100 --output_dir ./small_sample

# Include English data
python sampler/sample_fineweb2.py --total_docs 20000 --num_languages 50 --include_english
```

## Output

- JSONL file for each language
- Sampling statistics JSON file

## Remarks
the file fineweb2_meta.json can be used to sampel from all lanauges

the file filtered_fineweb2_meta.json only includes 190 lanaguges which are used in the most used evaluation dataset. In current training setups, we use this file to focus on these 190 langauges for now