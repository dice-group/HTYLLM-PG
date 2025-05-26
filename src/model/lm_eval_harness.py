#!/usr/bin/env python3
"""
LM Evaluation Harness script for custom GPT-2 model.

This script integrates our custom GPT-2 model with the LM Evaluation Harness library
to evaluate the model on various benchmarks and tasks.

Usage:
    python lm_eval_harness.py --model_path path/to/model.pt --tasks hellaswag,arc_easy,arc_challenge
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import sentencepiece as spm
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging

# Add the parent directory to the path to import our model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt_2_multi_gpu import GPT, GPTConfig

# Try to import lm_eval - if not available, provide installation instructions
try:
    import lm_eval
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
    from lm_eval.models.utils import Collator
    from lm_eval import evaluator
except ImportError:
    print("ERROR: lm_eval library not found!")
    print("Please install it with: pip install lm_eval")
    print("Or: pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomGPT2LM(LM):
    """
    Custom LM wrapper for our GPT-2 model to work with LM Evaluation Harness.
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str = "auto",
        batch_size: int = 1,
        max_length: int = 1024,
        **kwargs
    ):
        super().__init__()
        
        # Set device
        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device
            
        logger.info(f"Using device: {self._device}")
        
        # Load tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
        logger.info(f"Loaded tokenizer from {tokenizer_path}")
        logger.info(f"Vocabulary size: {self.tokenizer.get_piece_size()}")
        
        # Load model
        self._load_model(model_path)
        
        # Set parameters
        self._batch_size = batch_size
        self._max_length = max_length
        
        # Required attributes for LM Evaluation Harness
        self.vocab_size = self.tokenizer.get_piece_size()
        self.eot_token_id = self.tokenizer.eos_id() if hasattr(self.tokenizer, 'eos_id') else None
        
        # Additional required attributes
        self._max_length = max_length
        self._batch_size = batch_size
        
    def _load_model(self, model_path: str):
        """Load the GPT-2 model from checkpoint."""
        logger.info(f"Loading model from {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self._device)
        
        # Create model config - adjust these based on your model
        config = GPTConfig(
            vocab_size=self.tokenizer.get_piece_size(),
            block_size=1024,
            n_layer=24,  # Adjust based on your model
            n_head=16,   # Adjust based on your model  
            n_embd=1024, # Adjust based on your model
        )
        
        # Initialize model
        self.model = GPT(config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self._device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
        
    def tok_encode(self, string: str, **kwargs) -> List[int]:
        """Encode a string into tokens."""
        return self.tokenizer.encode(string)
    
    def tok_decode(self, tokens: List[int], **kwargs) -> str:
        """Decode tokens into a string."""
        return self.tokenizer.decode(tokens)
    
    def loglikelihood(self, requests: List[tuple]) -> List[tuple]:
        """
        Compute log-likelihood for a list of (context, continuation) pairs.
        
        Args:
            requests: List of Instance objects or (context, continuation) tuples
            
        Returns:
            List of (log_likelihood, is_greedy) tuples
        """
        results = []
        
        for request in requests:
            # Handle both Instance objects and tuples
            if hasattr(request, 'args'):
                # This is an Instance object from lm_eval
                context = request.args[0]
                continuation = request.args[1]
            else:
                # This is a tuple (for backward compatibility)
                context, continuation = request
            
            # Encode context and continuation
            context_tokens = self.tok_encode(context)
            continuation_tokens = self.tok_encode(continuation)
            
            # Combine context and continuation
            full_tokens = context_tokens + continuation_tokens
            
            if len(full_tokens) > self._max_length:
                # Truncate from the left if too long
                full_tokens = full_tokens[-self._max_length:]
                context_len = max(0, len(full_tokens) - len(continuation_tokens))
            else:
                context_len = len(context_tokens)
            
            # Convert to tensor
            input_ids = torch.tensor([full_tokens], dtype=torch.long, device=self._device)
            
            with torch.no_grad():
                logits, _ = self.model(input_ids)
                
            # Get logits for continuation tokens
            continuation_logits = logits[0, context_len-1:context_len-1+len(continuation_tokens)]
            continuation_targets = torch.tensor(continuation_tokens, device=self._device)
            
            # Compute log probabilities
            log_probs = F.log_softmax(continuation_logits, dim=-1)
            
            # Get log likelihood for each token in continuation
            token_log_likelihoods = log_probs.gather(1, continuation_targets.unsqueeze(1)).squeeze(1)
            
            # Sum log likelihoods
            total_log_likelihood = token_log_likelihoods.sum().item()
            
            # Check if this is the greedy choice
            greedy_tokens = continuation_logits.argmax(dim=-1)
            is_greedy = torch.equal(greedy_tokens, continuation_targets)
            
            results.append((total_log_likelihood, is_greedy))
            
        return results
    
    def loglikelihood_rolling(self, requests: List[str]) -> List[float]:
        """
        Compute rolling log-likelihood for a list of strings.
        
        Args:
            requests: List of Instance objects or strings
            
        Returns:
            List of log-likelihoods
        """
        results = []
        
        for request in requests:
            # Handle both Instance objects and strings
            if hasattr(request, 'args'):
                # This is an Instance object from lm_eval
                string = request.args[0]
            else:
                # This is a string (for backward compatibility)
                string = request
            
            tokens = self.tok_encode(string)
            
            if len(tokens) <= 1:
                results.append(0.0)
                continue
                
            if len(tokens) > self._max_length:
                tokens = tokens[-self._max_length:]
            
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self._device)
            
            with torch.no_grad():
                logits, _ = self.model(input_ids)
                
            # Compute log probabilities for all positions
            log_probs = F.log_softmax(logits[0], dim=-1)
            
            # Get log likelihood for each token (except the first)
            target_tokens = torch.tensor(tokens[1:], device=self._device)
            token_log_likelihoods = log_probs[:-1].gather(1, target_tokens.unsqueeze(1)).squeeze(1)
            
            # Sum all log likelihoods
            total_log_likelihood = token_log_likelihoods.sum().item()
            results.append(total_log_likelihood)
            
        return results
    
    def generate_until(self, requests: List[tuple]) -> List[str]:
        """
        Generate text until stopping criteria are met.
        
        Args:
            requests: List of Instance objects or (context, generation_kwargs) tuples
            
        Returns:
            List of generated strings
        """
        results = []
        
        for request in requests:
            # Handle both Instance objects and tuples
            if hasattr(request, 'args'):
                # This is an Instance object from lm_eval
                context = request.args[0]
                gen_kwargs = request.args[1] if len(request.args) > 1 else {}
            else:
                # This is a tuple (for backward compatibility)
                context, gen_kwargs = request
            
            # Parse generation parameters
            max_gen_toks = gen_kwargs.get("max_gen_toks", 50)
            until = gen_kwargs.get("until", [])
            temperature = gen_kwargs.get("temperature", 1.0)
            top_k = gen_kwargs.get("top_k", 50)
            
            # Encode context
            context_tokens = self.tok_encode(context)
            
            if len(context_tokens) > self._max_length - max_gen_toks:
                context_tokens = context_tokens[-(self._max_length - max_gen_toks):]
            
            input_ids = torch.tensor([context_tokens], dtype=torch.long, device=self._device)
            
            # Generate
            generated_tokens = []
            
            with torch.no_grad():
                for _ in range(max_gen_toks):
                    logits, _ = self.model(input_ids)
                    next_token_logits = logits[0, -1, :] / temperature
                    
                    # Apply top-k filtering
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                        probs = F.softmax(top_k_logits, dim=-1)
                        next_token_idx = torch.multinomial(probs, 1)
                        next_token = top_k_indices[next_token_idx].item()
                    else:
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, 1).item()
                    
                    generated_tokens.append(next_token)
                    
                    # Check stopping criteria
                    generated_text = self.tok_decode(generated_tokens)
                    if any(stop_seq in generated_text for stop_seq in until):
                        break
                    
                    # Update input for next iteration
                    input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=self._device)], dim=1)
                    
                    # Truncate if too long
                    if input_ids.size(1) > self._max_length:
                        input_ids = input_ids[:, 1:]
            
            # Decode generated tokens
            generated_text = self.tok_decode(generated_tokens)
            
            # Remove stopping sequences
            for stop_seq in until:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]
                    break
                    
            results.append(generated_text)
            
        return results

    @property
    def max_length(self):
        """Maximum sequence length the model can handle."""
        return self._max_length
    
    @property
    def max_gen_toks(self):
        """Maximum number of tokens to generate."""
        return 256  # Default value
    
    @property
    def batch_size(self):
        """Batch size for evaluation."""
        return self._batch_size
    
    @property
    def device(self):
        """Device the model is running on."""
        return self._device


# Register our model with lm_eval
@register_model("custom_gpt2")
class CustomGPT2LMEval(CustomGPT2LM):
    """Registered version of our custom GPT-2 model for lm_eval."""
    pass


def main():
    parser = argparse.ArgumentParser(description="Evaluate custom GPT-2 model using LM Evaluation Harness")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the model checkpoint (.pt file)"
    )
    
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="tokenizer/sp_model.model",
        help="Path to the SentencePiece tokenizer model"
    )
    
    parser.add_argument(
        "--tasks",
        type=str,
        default="hellaswag,arc_easy,arc_challenge,piqa,winogrande",
        help="Comma-separated list of tasks to evaluate on"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run evaluation on (auto, cuda, cpu)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples per task (for testing)"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="evaluation_results.json",
        help="Path to save evaluation results"
    )
    
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples"
    )
    
    parser.add_argument(
        "--log_samples",
        action="store_true",
        help="Log individual sample results"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        sys.exit(1)
        
    if not os.path.exists(args.tokenizer_path):
        logger.error(f"Tokenizer path does not exist: {args.tokenizer_path}")
        sys.exit(1)
    
    # Parse tasks
    tasks = [task.strip() for task in args.tasks.split(",")]
    logger.info(f"Evaluating on tasks: {tasks}")
    
    # Initialize model
    logger.info("Initializing model...")
    model = CustomGPT2LM(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Run evaluation
    logger.info("Starting evaluation...")
    
    results = evaluator.simple_evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        log_samples=args.log_samples,
    )
    
    # Save results
    logger.info(f"Saving results to {args.output_path}")
    
    import json
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("Evaluation completed!")
    logger.info("Results summary:")
    
    for task_name, task_results in results["results"].items():
        logger.info(f"\n{task_name}:")
        for metric_name, metric_value in task_results.items():
            if isinstance(metric_value, (int, float)):
                logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    # Print overall statistics if available
    if "groups" in results:
        logger.info("\nGroup averages:")
        for group_name, group_results in results["groups"].items():
            for metric_name, metric_value in group_results.items():
                if isinstance(metric_value, (int, float)):
                    logger.info(f"  {group_name} {metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main() 