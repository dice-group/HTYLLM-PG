# Usage: python print_jsonl_gz.py data.jsonl.gz
import gzip
import json
import sys

path = sys.argv[1]

with gzip.open(path, "rt") as f:
    for line in f:
        obj = json.loads(line)
        print(obj)
