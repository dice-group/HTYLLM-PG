#!/usr/bin/env python3
import gzip
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def extract_text_from_file(path):
    """
    Read a .jsonl.gz file, pull out the "text" field
    from each JSON line, and return a list of cleaned lines.
    """
    out_lines = []
    with gzip.open(path, 'rt', encoding='utf8') as f:
        for raw in f:
            try:
                j = json.loads(raw)
                # collapse any internal newlines and append newline
                out_lines.append(j["text"].replace("\n", " ") + "\n")
            except (json.JSONDecodeError, KeyError):
                # skip malformed lines
                continue
    return out_lines

def main():
    data_dir = Path(".")
    # find all .jsonl.gz files recursively
    files = list(data_dir.rglob("*.jsonl.gz"))
    print(f"Found {len(files)} files.")

    # open output once
    with open("corpus.txt", "w", encoding="utf8") as out_f:
        # pool of workers = number of CPUs
        with Pool(cpu_count()) as pool:
            # imap_unordered yields each file’s result as soon as it’s ready
            for extracted in tqdm(
                pool.imap_unordered(extract_text_from_file, files),
                total=len(files),
                desc="Extracting files",
                unit="file",
            ):
                out_f.writelines(extracted)

if __name__ == "__main__":
    main()
