import gzip
import os

def uncompressed_size_generator(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".jsonl.gz"):
                fpath = os.path.join(dirpath, fname)
                try:
                    with gzip.open(fpath, 'rb') as f:
                        yield sum(len(chunk) for chunk in iter(lambda: f.read(1024 * 1024), b''))
                except Exception as e:
                    print(f"Skipped {fpath}: {e}")

root = "/data/joel/language_adapters_subsets/arabic_subset/"
total_bytes = sum(uncompressed_size_generator(root))
print(f"Total uncompressed size: {total_bytes / (1024 ** 3):.2f} GB")
