#!/usr/bin/env python
"""
Standalone evaluation script for running lm-harness on a specific checkpoint.
Evaluates on hellaswag and belebele tasks.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import os
import sys

# Set datasets cache to local directory to avoid permission issues
cache_dir = Path("./cache/huggingface_datasets").absolute()
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
os.environ["DATASETS_CACHE"] = str(cache_dir)  
os.environ["HF_HOME"] = str(cache_dir.parent)

import torch
from transformers import AutoTokenizer
from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM


def setup_args():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on hellaswag and belebele")
    
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        required=True,
        help="Path to the model checkpoint directory"
    )
    
    parser.add_argument(
        "--tokenizer_path", 
        type=str, 
        default="tokenizer",
        help="Path to the tokenizer directory (default: tokenizer)"
    )
    
    parser.add_argument(
        "--tasks", 
        type=str, 
        nargs="+",
        default=["hellaswag", "belebele"],
        help="Tasks to evaluate on (default: hellaswag belebele)"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16,
        help="Batch size for evaluation (default: 16)"
    )
    
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit number of examples per task (default: None - use all)"
    )
    
    parser.add_argument(
        "--fewshot", 
        type=int, 
        default=0,
        help="Number of fewshot examples (default: 0)"
    )
    
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=None,
        help="Path to save evaluation results (default: results_<checkpoint_name>.json)"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use: 'auto', 'cuda', 'cpu' (default: auto)"
    )
    
    return parser.parse_args()


def load_model_and_tokenizer(checkpoint_path: str, tokenizer_path: str, device: str):
    """Load tokenizer from the specified path."""
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Will load model from: {checkpoint_path} on device: {device}")
    return checkpoint_path, tokenizer, device


def run_evaluation(checkpoint_path, tokenizer, tasks_list, batch_size, limit, fewshot, device):
    """Run lm-eval harness evaluation."""
    print(f"Setting up evaluation for tasks: {tasks_list}")
    
    # Let HFLM handle model loading - more efficient and uses HFLM optimizations
    # Use both GPUs with model parallelism
    lm = HFLM(
        pretrained=checkpoint_path,  # Pass path as string
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        dtype="auto",  # Let HFLM choose the best dtype
        trust_remote_code=False,
        parallelize=True,  # Enable model parallelism across GPUs
        device_map="auto",  # Automatically distribute model across GPUs
    )
    
    print("Starting evaluation...")
    
    # Run evaluation (note: fewshot is handled by task configuration in 0.4.x)
    results = evaluator.evaluate(
        lm,
        task_dict=tasks.get_task_dict(tasks_list),
        limit=limit,
    )
    
    return results


def save_results(results, output_path):
    """Save evaluation results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


def print_results(results):
    """Print a summary of evaluation results."""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    for task_name, task_results in results["results"].items():
        print(f"\nTask: {task_name}")
        print("-" * 30)
        
        # Print key metrics
        for metric_name, metric_value in task_results.items():
            if isinstance(metric_value, (int, float)):
                print(f"  {metric_name}: {metric_value:.4f}")
            else:
                print(f"  {metric_name}: {metric_value}")


def main():
    args = setup_args()
    
    # Validate checkpoint path
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint path does not exist: {checkpoint_path}")
        sys.exit(1)
    
    # Validate tokenizer path
    tokenizer_path = Path(args.tokenizer_path)
    if not tokenizer_path.exists():
        print(f"Error: Tokenizer path does not exist: {tokenizer_path}")
        sys.exit(1)
    
    print(f"Evaluating checkpoint: {checkpoint_path}")
    print(f"Tasks: {args.tasks}")
    print(f"Batch size: {args.batch_size}")
    print(f"Limit: {args.limit}")
    print(f"Fewshot: {args.fewshot}")
    
    try:
        # Load model and tokenizer
        checkpoint_path, tokenizer, device = load_model_and_tokenizer(
            str(checkpoint_path), 
            str(tokenizer_path), 
            args.device
        )
        
        # Run evaluation
        results = run_evaluation(
            checkpoint_path, 
            tokenizer, 
            args.tasks, 
            args.batch_size, 
            args.limit, 
            args.fewshot, 
            device
        )
        
        # Print results
        print_results(results)
        
        # Save results
        if args.output_path is None:
            checkpoint_name = checkpoint_path.name
            output_path = f"results_{checkpoint_name}.json"
        else:
            output_path = args.output_path
            
        save_results(results, output_path)
        
        print(f"\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 