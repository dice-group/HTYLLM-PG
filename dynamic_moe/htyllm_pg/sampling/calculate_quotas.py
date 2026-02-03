import json
import numpy as np
import pandas as pd
import argparse

def calculate_sampling_quotas(inventory_path="dataset_inventory.json", total_cap_gb=100.0, num_langs=None):
    with open(inventory_path, 'r') as f:
        data = json.load(f)
        
    # 1. Aggregate statistics
    langs = []
    sizes = []
    sources = [] # 'fw2' or 'fw1'
    
    # FineWeb-2
    for lang, info in data["fineweb-2"].items():
        langs.append(lang)
        sizes.append(info["total_size"])
        sources.append("fw2")
        
    # FineWeb-1 (English)
    langs.append("eng_Latn") # Use standard code
    sizes.append(data["fineweb-1"]["total_size"])
    sources.append("fw1")
    
    df = pd.DataFrame({"lang": langs, "size_bytes": sizes, "source": sources})
    
    # Filter top N languages if requested
    if num_langs is not None:
        print(f"Selecting top {num_langs} languages by size...")
        df = df.sort_values("size_bytes", ascending=False).head(num_langs)
        # Reset index to ensure loops work correctly
        df = df.reset_index(drop=True)
    
    # 2. Water-Filling Algorithm (Maximize Minimum Allocation)
    # We want to distribute TotalCap such that we are as close to Uniform(TotalCap/N) as possible,
    # constrained by Size_i.
    
    total_cap_bytes = total_cap_gb * 1e9
    remaining_budget = total_cap_bytes
    
    # Initialize allocations
    df["final_bytes"] = 0.0
    
    # Sort by available size (smallest first) to easily check saturation
    df_sorted = df.sort_values("size_bytes", ascending=True)
    # Get indices of sorted dataframe to map back to original df
    sorted_indices = df_sorted.index.tolist()
    
    n = len(df_sorted)
    
    for i in range(n):
        # Remaining languages to consider
        remaining_langs = n - i
        
        # Calculate fair share for remaining languages
        fair_share = remaining_budget / remaining_langs
        
        # Get current language stats
        lang_idx = sorted_indices[i]
        current_size = df.at[lang_idx, "size_bytes"]
        
        if current_size < fair_share:
            # This language cannot meet the fair share. Take it all.
            df.at[lang_idx, "final_bytes"] = current_size
            remaining_budget -= current_size
        else:
            # This language (and all subsequent ones, since sorted) can meet the fair share.
            # Distribute remaining budget equally among them.
            for j in range(i, n):
                other_idx = sorted_indices[j]
                df.at[other_idx, "final_bytes"] = fair_share
            
            remaining_budget = 0
            break
            
    print(f"Total Cap: {total_cap_gb} GB")
    if num_langs:
        print(f"Languages Selected: {len(df)}")
        
    scheduled_gb = df['final_bytes'].sum() / 1e9
    print(f"Scheduled: {scheduled_gb:.2f} GB")
    
    if abs(scheduled_gb - total_cap_gb) > 1.0 and remaining_budget > 1e9:
         print(f"Warning: Could not utilize full capacity. Unused: {remaining_budget/1e9:.2f} GB")
         
    print("-" * 20)
    print("Top 10 allocations:")
    print(df.sort_values("final_bytes", ascending=False).head(10)[["lang", "source", "size_bytes", "final_bytes"]])
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap_gb", type=float, default=100.0, help="Total data cap in GB")
    parser.add_argument("--num_langs", type=int, default=None, help="Number of top languages to select")
    parser.add_argument("--inventory", type=str, default="dataset_inventory.json")
    parser.add_argument("--output", type=str, default="sampling_quotas.csv")
    args = parser.parse_args()

    df = calculate_sampling_quotas(args.inventory, args.cap_gb, args.num_langs)
    df.to_csv(args.output, index=False)
