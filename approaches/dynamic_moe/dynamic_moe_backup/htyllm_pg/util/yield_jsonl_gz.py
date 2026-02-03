# Usage: for batch in yield_jsonl_gz('/path/to/folder', batch_size=1000): ...

import gzip
import json
from pathlib import Path

def yield_jsonl_gz(folder_path, batch_size=1000):
    batch = []
    for file_path in sorted(Path(folder_path).glob('*.jsonl.gz')):
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get('text', '')
                if text:
                    batch.append(text)
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
    if batch:
        yield batch


if __name__ == '__main__':
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else '.'
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    
    for i, batch in enumerate(yield_jsonl_gz(folder, batch_size)):
        if i >= 5:
            break
        print(f"Batch {i+1}: {len(batch)} texts")
        print(f"First text preview: {batch[0]}")
        print()

