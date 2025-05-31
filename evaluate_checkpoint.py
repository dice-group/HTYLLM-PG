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


class SafeHFLM(HFLM):
    """HFLM wrapper that handles empty continuation strings gracefully."""
    
    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        """Override to handle empty continuation tokens."""
        # Patch requests to handle empty continuations
        safe_requests = []
        for req in requests:
            context_enc, continuation_enc = req.args
            
            # If continuation is empty, substitute with space token
            if len(continuation_enc) == 0:
                # Use space token as fallback
                space_enc = self.tokenizer.encode(" ", add_special_tokens=False)
                if len(space_enc) > 0:
                    continuation_enc = space_enc[:1]  # Use just the first token
                else:
                    # Fallback to UNK token if space also fails
                    continuation_enc = [self.tokenizer.unk_token_id]
                
                # Create new request with safe continuation
                from lm_eval.api.instance import Instance
                safe_req = Instance(
                    request_type=req.request_type,
                    doc=req.doc,
                    arguments=(context_enc, continuation_enc),
                    idx=req.idx,
                    metadata=req.metadata
                )
                safe_requests.append(safe_req)
            else:
                safe_requests.append(req)
        
        # Call parent with safe requests
        return super()._loglikelihood_tokens(safe_requests, disable_tqdm=disable_tqdm)


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
    
    # Don't override tokenizer settings - it already has proper special tokens
    print(f"Tokenizer special tokens:")
    print(f"  BOS: {tokenizer.bos_token}")
    print(f"  EOS: {tokenizer.eos_token}")
    print(f"  PAD: {tokenizer.pad_token}")
    print(f"  UNK: {tokenizer.unk_token}")
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Will load model from: {checkpoint_path} on device: {device}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    return checkpoint_path, tokenizer, device


def run_evaluation(checkpoint_path, tokenizer, tasks_list, batch_size, limit, fewshot, device):
    """Run lm-eval harness evaluation."""
    print(f"Setting up evaluation for tasks: {tasks_list}")
    
    # Use both GPUs with model parallelism + safe empty continuation handling
    lm = SafeHFLM(
        pretrained=checkpoint_path,  # Pass path as string
        tokenizer=checkpoint_path,   # Use tokenizer from checkpoint, not separate dir
        device=device,
        batch_size=batch_size,  # Use original batch size
        dtype="auto",  # Let HFLM choose the best dtype
        trust_remote_code=False,
        parallelize=True,  # Enable model parallelism across GPUs
        device_map="auto",  # Automatically distribute model across GPUs
        add_bos_token=False,  # Don't add BOS token automatically
        truncation=True,  # Enable truncation for safety
    )
    
    print("Starting evaluation...")
    print(f"Model tokenizer vocab size: {lm.tokenizer.vocab_size}")
    print(f"Model tokenizer type: {type(lm.tokenizer)}")
    print(f"Model tokenizer special tokens:")
    print(f"  BOS: {lm.tokenizer.bos_token}")
    print(f"  EOS: {lm.tokenizer.eos_token}")  
    print(f"  PAD: {lm.tokenizer.pad_token}")
    print(f"  UNK: {lm.tokenizer.unk_token}")
    
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