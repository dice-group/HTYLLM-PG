import os
import json
import time
from pathlib import Path
from huggingface_hub import list_repo_tree

CACHE_FILE = "dataset_inventory.json"

def get_dataset_inventory():
    if os.path.exists(CACHE_FILE):
        print(f"Loading inventory from {CACHE_FILE}...")
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)

    inventory = {
        "fineweb-2": {}, # lang_code -> { "total_size": int, "files": [ {path, size} ] }
        "fineweb-1": { "total_size": 0, "files": [] } # Treat as one "en" bucket
    }

    print("Fetching FineWeb-2 file list...")
    fw2_files = list(list_repo_tree("HuggingFaceFW/fineweb-2", recursive=True, repo_type="dataset"))
    
    for item in fw2_files:
        if not hasattr(item, 'path') or not item.path.endswith('.parquet'):
            continue
        
        # Structure: data/<lang_code>/train/*.parquet
        parts = item.path.split('/')
        if len(parts) >= 3 and parts[0] == 'data' and parts[2] == 'train' and not parts[1].endswith('_removed'):
            lang = parts[1]
            if lang not in inventory["fineweb-2"]:
                inventory["fineweb-2"][lang] = {"total_size": 0, "files": []}
            
            size = getattr(item, 'size', 0)
            inventory["fineweb-2"][lang]["files"].append({"path": item.path, "size": size})
            inventory["fineweb-2"][lang]["total_size"] += size

    print("Fetching FineWeb-1 (English) file list...")
    # FineWeb-1 is huge. Listing all might take too long/be too much.
    fw1_files = list(list_repo_tree("HuggingFaceFW/fineweb", recursive=True, repo_type="dataset"))
    
    for item in fw1_files:
        if not hasattr(item, 'path') or not item.path.endswith('.parquet'):
            continue
            
        # Structure: data/<dump>/...
        # We treat all as "en"
        size = getattr(item, 'size', 0)
        inventory["fineweb-1"]["files"].append({"path": item.path, "size": size})
        inventory["fineweb-1"]["total_size"] += size

    print(f"Saving inventory to {CACHE_FILE}...")
    with open(CACHE_FILE, 'w') as f:
        json.dump(inventory, f)
        
    return inventory

if __name__ == "__main__":
    inv = get_dataset_inventory()
    print(f"FineWeb-2 Languages: {len(inv['fineweb-2'])}")
    print(f"FineWeb-1 Total Size: {inv['fineweb-1']['total_size'] / 1e9:.2f} GB")

