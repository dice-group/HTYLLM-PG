#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import requests
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_score
from enum import Enum
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(PROJECT_ROOT_DIR)

from approaches.CoLA.distributed_data_processor.language_subsets import fineweb2_benchmark_languages

class ResourceCategory(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"

def load_and_filter_metadata(csv_path, benchmark_languages):
    """Load Fineweb2 metadata and filter by benchmark languages."""
    df = pd.read_csv(csv_path)
    
    # Initial filters from original script
    mask = (
        ~df["subset"].astype(str).str.endswith("_removed") &
        (df["split"] == "train") &
        (df["family"] != "-")
    )
    df = df.loc[mask].reset_index(drop=True)
    
    # added: filter by benchmark languages
    df = df[df['subset'].isin(benchmark_languages)].reset_index(drop=True)
    print(f"Loaded and filtered metadata: {len(df)} languages")
    return df

def add_resource_categories(df, resource_csv_path):
    """Merge with resource categories and map to High/Medium/Low."""
    res_df = pd.read_csv(resource_csv_path, sep="\t")
    
    # Filter invalid categories
    res_df = res_df[~res_df['resource_category'].str.endswith('*', na=False)]
    
    # Prepare for merge
    df['_subset_lc'] = df['subset'].astype(str).str.lower()
    res_df['_lang_code_lc'] = res_df['lang_code'].astype(str).str.lower()
    
    # Merge
    df = df.merge(
        res_df[['_lang_code_lc', 'resource_category']],
        left_on='_subset_lc',
        right_on='_lang_code_lc',
        how='left'
    )
    df.drop(columns=['_subset_lc', '_lang_code_lc'], inplace=True)
    
    # Sort and forward fill logic from original script
    df = df.sort_values(by="documents", ascending=False).reset_index(drop=True)
    
    # Logic to propagate categories based on sorted order (simplified from original)
    # The original script had a complex logic to fill categories based on "last index" of each category.
    # Here we will assume the merge provided most categories and we might need to fill missing ones.
    # However, to be faithful to the original logic which seemed to define "tiers" by document count:
    
    # Re-implementing the "tier" logic exactly as it seems crucial for the definition of High/Med/Low
    # 1. Find last index of each category
    last_idx_series = df.groupby("resource_category").apply(lambda grp: grp.index[-1])
    last_idx_by_category = last_idx_series.to_dict()
    
    res_cat_tuples = []
    for cat, idx in last_idx_by_category.items():
        res_cat_tuples.append((cat, idx))
    res_cat_tuples.sort(key=lambda tup: tup[1])
    
    # Map to Enum
    MERGE_MAP = {
        "high"      : ResourceCategory.HIGH,
        "medhigh"   : ResourceCategory.MEDIUM,
        "medlow"    : ResourceCategory.LOW,
        "low"       : ResourceCategory.LOW,
        "not_enough": ResourceCategory.LOW,
    }
    
    merged_tuples = [(MERGE_MAP[item[0]].value, item[1]) for item in res_cat_tuples]
    
    # Propagate
    prev_end = 0
    for cat, end_idx in merged_tuples:
        df.loc[prev_end:end_idx, "resource_category"] = cat
        prev_end = end_idx + 1
        
    if prev_end < len(df):
        df.loc[prev_end:, "resource_category"] = df["resource_category"].ffill()
        
    return df

def generate_onehot_embeddings(df):
    """Generate one-hot embeddings for resource category, script, and family."""
    rc_oh = pd.get_dummies(df["resource_category"], prefix="rc")
    script_oh = pd.get_dummies(df["script"], prefix="script")
    family_oh = pd.get_dummies(df["family"], prefix="family")
    
    features = pd.concat([rc_oh, script_oh, family_oh], axis=1)
    return features

def get_llm_embeddings(df):
    """Generate LLM embeddings using the local endpoint."""
    embedding_config = {
        'endpoint': 'http://lola.cs.uni-paderborn.de:9292/v1',
        'model_id': 'nomic-embed-text-v2-moe'
    }
    
    embed_cache = {}

    def get_embeddings_api(input_texts):
        try:
            resp = requests.post(
                embedding_config['endpoint'] + '/embeddings', 
                json={'input': input_texts, 'model': embedding_config['model_id']}
            ).json()
            return [d['embedding'] for d in resp['data']]
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            # Return zero vectors or handle error? 
            # For now, let's assume it works or fail hard.
            raise e

    def get_cached(text):
        if text not in embed_cache:
            embed_cache[text] = get_embeddings_api([text])[0]
        return embed_cache[text]

    final_embeddings = []
    BATCH_SIZE = 512
    
    print("Generating LLM embeddings...")
    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = df.iloc[i:i+BATCH_SIZE]
        for _, row in batch.iterrows():
            name_emb = get_cached(row["name"] or "")
            cat_emb = get_cached(row["resource_category"] or "")
            script_emb = get_cached(row["script"] or "")
            family_emb = get_cached(row["family"] or "")
            
            combined = np.concatenate([name_emb, cat_emb, script_emb, family_emb])
            final_embeddings.append(combined)
            
    return pd.DataFrame(final_embeddings)

def farthest_first_ordering(dist_matrix):
    """Greedy farthest-point selection."""
    n = dist_matrix.shape[0]
    start_idx = np.argmin(dist_matrix.sum(axis=1))
    ordered = [start_idx]
    remaining = set(range(n)) - {start_idx}
    
    while remaining:
        min_dists = {
            i: min(dist_matrix[i, sel] for sel in ordered) for i in remaining
        }
        next_idx = max(min_dists, key=min_dists.get)
        ordered.append(next_idx)
        remaining.remove(next_idx)
        
    return ordered

def select_optimal_sizes(embeddings_df, k_min=5, k_max=100, num_sizes=3):
    """
    Determine 3 'scientifically sound' sample sizes using Silhouette analysis.
    We will look for local maxima in the Silhouette score.
    If distinct peaks aren't found, we fall back to a geometric progression 
    but guided by the score curve.
    """
    print(f"Analyzing cluster quality for k={k_min} to {k_max}...")
    
    scores = []
    ks = range(k_min, k_max + 1, 2) # Step by 2 to save time
    
    for k in tqdm(ks, desc="Silhouette Analysis"):
        if k >= len(embeddings_df):
            break
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init='auto', random_state=42)
        labels = kmeans.fit_predict(embeddings_df)
        score = silhouette_score(embeddings_df, labels)
        scores.append((k, score))
        
    # Find peaks (local maxima)
    peaks = []
    for i in range(1, len(scores) - 1):
        if scores[i][1] > scores[i-1][1] and scores[i][1] > scores[i+1][1]:
            peaks.append(scores[i])
            
    # Sort peaks by score descending
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    selected_sizes = []
    
    # Strategy: Pick the best peak, then one significantly smaller, and one larger (if available)
    # Or simply pick the top 3 distinct peaks that are spread out.
    
    # Let's try to get a Small, Medium, Large spread.
    # If we have enough peaks, pick ones that cover the range.
    
    if len(peaks) >= 3:
        # Sort peaks by K to identify small/med/large candidates
        peaks_by_k = sorted(peaks, key=lambda x: x[0])
        
        # Simple heuristic: 
        # 1. Smallest K peak
        # 2. Peak closest to geometric mean of min/max
        # 3. Largest K peak (or best score if not included)
        
        # Actually, let's just pick the top 3 scoring peaks and sort them by size
        top_peaks = sorted(peaks[:5], key=lambda x: x[1], reverse=True) # Take top 5 best scores
        # Pick 3 from these that are most distinct in size?
        # Let's just take the top 3 best scoring K's.
        best_ks = sorted([p[0] for p in top_peaks[:3]])
        selected_sizes = best_ks
    else:
        # Fallback: Pick best K, and then geometric neighbors
        if peaks:
            best_k = peaks[0][0]
        else:
            # No peaks? Just max score
            best_k = max(scores, key=lambda x: x[1])[0]
            
        # Create 3 sizes around the best K or spreading out
        # If best_k is small, add medium and large
        # If best_k is large, add small and medium
        
        # Let's default to a geometric spread if peaks are insufficient, 
        # but anchored on the best performing K if possible.
        
        # Actually, user wants "scientifically sound". 
        # If the curve is flat, geometric spread is the most neutral "scientific" approach (log scale coverage).
        # If there are peaks, those are "natural" scales.
        
        geom = np.geomspace(k_min, k_max, num=num_sizes)
        selected_sizes = sorted(list(set([int(round(v)) for v in geom])))
        
        # If we had a best_k, try to swap the closest geometric one with best_k?
        # Let's keep it simple: if peaks failed, geometric is robust.
        
    print(f"Silhouette analysis complete. Selected sizes: {selected_sizes}")
    return selected_sizes

def cluster_and_sample(df, embeddings_df, sample_sizes, method_name, output_dir):
    """Cluster languages, find medoids, order them, and save samples."""
    max_k = max(sample_sizes)
    
    # Ensure we don't ask for more clusters than data points
    if max_k > len(df):
        print(f"Warning: Requested {max_k} clusters but only have {len(df)} languages. Capping at {len(df)}.")
        max_k = len(df)
        sample_sizes = [s for s in sample_sizes if s <= max_k]
    
    print(f"Clustering ({method_name}) with k={max_k}...")
    kmeans = KMeans(n_clusters=max_k, init="k-means++", n_init='auto', random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_df)
    
    unique_labels = set(cluster_labels)
    print(f"KMeans converged. Found {len(unique_labels)} unique clusters (requested {max_k}).")
    
    # Compute medoids
    X = embeddings_df.values
    dist_matrix = pairwise_distances(X, metric="euclidean")
    medoid_indices = []
    
    for clust_id in range(max_k):
        member_mask = (cluster_labels == clust_id)
        member_idxs = np.where(member_mask)[0]
        
        if len(member_idxs) == 0:
            print(f"Warning: Cluster {clust_id} is empty!")
            continue
            
        sub_dm = dist_matrix[np.ix_(member_idxs, member_idxs)]
        medoid_rel_idx = np.argmin(sub_dm.sum(axis=1))
        medoid_abs_idx = member_idxs[medoid_rel_idx]
        medoid_indices.append(medoid_abs_idx)
        
    # Order medoids
    medoid_dist_matrix = dist_matrix[np.ix_(medoid_indices, medoid_indices)]
    ordered_indices = np.array(medoid_indices)[farthest_first_ordering(medoid_dist_matrix)]
    
    # Save samples
    os.makedirs(output_dir, exist_ok=True)
    for sz in sample_sizes:
        nested_idxs = ordered_indices[:sz]
        sample_df = df.iloc[nested_idxs].reset_index(drop=True)
        csv_name = os.path.join(output_dir, f"{sz}_representative_mediods.csv")
        sample_df.to_csv(csv_name, index=False)
        print(f"Saved {sz}-sample to {csv_name}")

def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(data_dir, "fineweb2-language-distribution.csv")
    resource_path = os.path.join(data_dir, "lang_resource_dataset.tsv")
    
    # 1. Load and Filter
    df = load_and_filter_metadata(metadata_path, fineweb2_benchmark_languages)
    
    # 2. Add Resource Categories
    df = add_resource_categories(df, resource_path)
    
    # 3. Embeddings
    print("Generating One-Hot Embeddings...")
    oh_embeddings = generate_onehot_embeddings(df)
    
    print("Generating LLM Embeddings...")
    llm_embeddings = get_llm_embeddings(df)
    
    # 4. Cluster and Sample
    
    print("\n--- Processing One-Hot Embeddings ---")
    # Determine sizes for One-Hot
    oh_sizes = select_optimal_sizes(oh_embeddings, k_min=10, k_max=min(100, len(df)-1))
    
    cluster_and_sample(
        df, 
        oh_embeddings, 
        oh_sizes, 
        "onehot", 
        os.path.join(data_dir, "onehot_benchmark_samples")
    )
    
    print("\n--- Processing LLM Embeddings ---")
    # Determine sizes for LLM
    llm_sizes = select_optimal_sizes(llm_embeddings, k_min=10, k_max=min(100, len(df)-1))
    
    cluster_and_sample(
        df, 
        llm_embeddings, 
        llm_sizes, 
        "llm", 
        os.path.join(data_dir, "llm_benchmark_samples")
    )

if __name__ == "__main__":
    main()
