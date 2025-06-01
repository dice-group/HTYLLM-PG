#!/usr/bin/env python3
"""
LM Evaluation Harness script for a custom GPT-2 model.

This script integrates a custom GPT-2 model with the LM Evaluation Harness library
to evaluate the model on various benchmarks and tasks.

Usage:
    python lm_eval_harness.py --model_path path/to/model.pt --tasks hellaswag,arc_easy,arc_challenge
"""

import os
import sys
import argparse
import logging

import torch
import torch.nn.functional as F
import sentencepiece as spm

# Ensure parent directory is on the path to import our GPT code
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple, Union, Any

from gpt_2_multi_gpu import GPT, GPTConfig

# Try to import lm_eval; exit with instructions if missing
try:
    import lm_eval
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
    from lm_eval import evaluator
except ImportError:
    print("ERROR: lm_eval library not found!")
    print("Please install it with: pip install lm_eval")
    print("Or: pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git")
    sys.exit(1)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomGPT2LM(LM):
    """
    Custom LM wrapper for a GPT-2 model to work with the LM Evaluation Harness.
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

        # Determine device
        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device
        logger.info(f"Using device: {self._device}")

        # Load SentencePiece tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        try:
            self.tokenizer.load(tokenizer_path)
        except Exception as e:
            logger.error(f"Failed to load tokenizer from {tokenizer_path}: {e}")
            sys.exit(1)
        vocab_size = self.tokenizer.get_piece_size()
        logger.info(f"Loaded tokenizer (vocab size = {vocab_size}) from {tokenizer_path}")

        # Store batch size and max sequence length
        self._batch_size = batch_size
        self._max_length = max_length

        # Load the GPT-2 model
        self._load_model(model_path, vocab_size)

        # Attributes required by LM Evaluation Harness
        self.vocab_size = vocab_size
        self.eot_token_id = (
            self.tokenizer.eos_id() if hasattr(self.tokenizer, "eos_id") else None
        )

    def _load_model(self, model_path: str, vocab_size: int) -> None:
        """Load the GPT-2 model from checkpoint."""
        if not os.path.exists(model_path):
            logger.error(f"Model path does not exist: {model_path}")
            sys.exit(1)
        logger.info(f"Loading model from {model_path}")

        # Load checkpoint
        try:
            checkpoint = torch.load(model_path, map_location=self._device)
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            sys.exit(1)

        # Build GPTConfig; adjust layers, heads, and embedding dims if needed
        config = GPTConfig(
            vocab_size=vocab_size,
            block_size=self._max_length,
            n_layer=24,
            n_head=16,
            n_embd=1024,
        )

        # Initialize model and load weights
        self.model = GPT(config)
        try:
            state_dict = checkpoint["model_state_dict"]
            
            # Remove _orig_mod. prefix if present (from torch.compile or similar wrappers)
            if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
                logger.info("Removing _orig_mod. prefix from state dict keys")
                state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}
            
            self.model.load_state_dict(state_dict)
        except KeyError:
            logger.error("Checkpoint does not contain 'model_state_dict'")
            sys.exit(1)
        self.model.to(self._device)
        self.model.eval()
        logger.info("Model loaded and set to eval mode")

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        """Encode a string into a list of token IDs."""
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens: List[int], **kwargs) -> str:
        """Decode a list of token IDs back into a string."""
        return self.tokenizer.decode(tokens)

    def loglikelihood(
        self, requests: List[Union[Tuple[str, str], Any]]
    ) -> List[Tuple[float, bool]]:
        """
        Compute (log_likelihood, is_greedy) for each (context, continuation) pair.

        Args:
            requests: A list of 2-tuples (context, continuation), or objects with an 'args' attribute.
        Returns:
            A list of tuples: (total_log_likelihood, is_greedy_match).
        """
        results: List[Tuple[float, bool]] = []

        for req in requests:
            # Unpack either an object with .args or a simple (context, continuation) tuple
            if hasattr(req, "args"):
                context, continuation = req.args[0], req.args[1]
            else:
                context, continuation = req  # type: ignore

            # Tokenize
            context_tokens = self.tok_encode(context)
            continuation_tokens = self.tok_encode(continuation)

            # Combine context + continuation for a single forward pass
            full_tokens = context_tokens + continuation_tokens
            if len(full_tokens) > self._max_length:
                # Truncate from the left
                full_tokens = full_tokens[-self._max_length :]
                context_len = max(0, len(full_tokens) - len(continuation_tokens))
            else:
                context_len = len(context_tokens)

            input_ids = torch.tensor([full_tokens], dtype=torch.long, device=self._device)

            with torch.no_grad():
                logits, _ = self.model(input_ids)  # (1, seq_len, vocab_size)

            # Extract logits for the continuation tokens
            if context_len == 0:
                cont_logits = logits[0, : len(continuation_tokens), :]
            else:
                start_idx = context_len - 1
                end_idx = start_idx + len(continuation_tokens)
                cont_logits = logits[0, start_idx:end_idx, :]

            target_ids = torch.tensor(continuation_tokens, device=self._device)

            # Compute log probabilities
            log_probs = F.log_softmax(cont_logits, dim=-1)  # (L, vocab_size)
            token_ll = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)  # (L,)

            total_ll = token_ll.sum().item()

            # Check if greedy decoding would match the continuation exactly
            greedy_ids = cont_logits.argmax(dim=-1)  # (L,)
            is_greedy = bool(torch.equal(greedy_ids, target_ids))

            results.append((total_ll, is_greedy))

        return results

    def loglikelihood_rolling(
        self, requests: List[Union[str, Any]]
    ) -> List[float]:
        """
        Compute rolling log-likelihood for each input string.

        Args:
            requests: A list of strings or objects with an 'args' attribute.
        Returns:
            List of log-likelihood values.
        """
        results: List[float] = []

        for req in requests:
            if hasattr(req, "args"):
                text = req.args[0]
            else:
                text = req  # type: ignore

            tokens = self.tok_encode(text)

            # If there's only one token (or none), rolling LL is zero
            if len(tokens) <= 1:
                results.append(0.0)
                continue

            # Truncate if too long
            if len(tokens) > self._max_length:
                tokens = tokens[-self._max_length :]

            input_ids = torch.tensor([tokens], dtype=torch.long, device=self._device)
            with torch.no_grad():
                logits, _ = self.model(input_ids)  # (1, seq_len, vocab_size)

            log_probs = F.log_softmax(logits[0], dim=-1)  # (seq_len, vocab_size)
            target_ids = torch.tensor(tokens[1:], device=self._device)  # (seq_len - 1,)

            token_ll = log_probs[:-1].gather(1, target_ids.unsqueeze(1)).squeeze(1)
            total_ll = token_ll.sum().item()
            results.append(total_ll)

        return results

    def generate_until(
        self, requests: List[Union[Tuple[str, dict], Any]]
    ) -> List[str]:
        """
        Generate text until specified stopping criteria are met.

        Args:
            requests: List of (context, generation_kwargs) pairs or objects with an 'args' attribute.
        Returns:
            List of generated strings.
        """
        results: List[str] = []

        for req in requests:
            if hasattr(req, "args"):
                context = req.args[0]
                gen_kwargs = req.args[1] if len(req.args) > 1 else {}
            else:
                context, gen_kwargs = req  # type: ignore

            max_gen_toks = gen_kwargs.get("max_gen_toks", 50)
            stop_seqs = gen_kwargs.get("until", [])
            temperature = gen_kwargs.get("temperature", 1.0)
            top_k = gen_kwargs.get("top_k", 50)

            context_tokens = self.tok_encode(context)
            if len(context_tokens) > self._max_length - max_gen_toks:
                context_tokens = context_tokens[-(self._max_length - max_gen_toks) :]

            input_ids = torch.tensor([context_tokens], dtype=torch.long, device=self._device)
            generated_tokens: List[int] = []

            with torch.no_grad():
                for _ in range(max_gen_toks):
                    logits, _ = self.model(input_ids)  # (1, seq_len, vocab)
                    next_logits = logits[0, -1, :] / temperature  # (vocab,)

                    if top_k > 0:
                        topk_vals, topk_idx = torch.topk(next_logits, top_k)
                        probs = F.softmax(topk_vals, dim=-1)
                        choice = torch.multinomial(probs, num_samples=1).item()
                        next_token = topk_idx[choice].item()
                    else:
                        probs = F.softmax(next_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).item()

                    generated_tokens.append(next_token)

                    # Check if any stop sequence has been generated
                    partial = self.tok_decode(generated_tokens)
                    if any(stop in partial for stop in stop_seqs):
                        # Truncate output before the stop sequence
                        for stop in stop_seqs:
                            if stop in partial:
                                partial = partial.split(stop)[0]
                                break
                        generated = partial
                        break

                    next_id = torch.tensor([[next_token]], device=self._device)
                    input_ids = torch.cat([input_ids, next_id], dim=1)

                    if input_ids.size(1) > self._max_length:
                        input_ids = input_ids[:, 1:]

                else:
                    # If loop completes without hitting a stop sequence
                    generated = self.tok_decode(generated_tokens)

            results.append(generated)

        return results

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> str:
        return self._device


# Register our model with lm_eval under the name "custom_gpt2"
@register_model("custom_gpt2")
class CustomGPT2LMEval(CustomGPT2LM):
    """Registered variant of CustomGPT2LM for lm_eval."""
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a custom GPT-2 model using LM Evaluation Harness"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="tokenizer/sp_model.model",
        help="Path to the SentencePiece tokenizer model",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="hellaswag,arc_easy,arc_challenge,piqa,winogrande",
        help="Comma-separated list of tasks to evaluate on",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run evaluation on (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples per task (for debugging)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="evaluation_results.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples per prompt",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        help="Log individual sample results to stdout",
    )

    args = parser.parse_args()

    # Parse task list
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        logger.error("No tasks specified for evaluation.")
        sys.exit(1)
    logger.info(f"Evaluating on tasks: {tasks}")

    # Initialize model
    logger.info("Initializing CustomGPT2LM...")
    model = CustomGPT2LM(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        batch_size=args.batch_size,
        max_length=1024,
    )

    # Run evaluation
    logger.info("Starting evaluation...")
    try:
        results = evaluator.simple_evaluate(
            model=model,
            tasks=tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            limit=args.limit,
            log_samples=args.log_samples,
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

    # Save results to JSON
    import json

    try:
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved evaluation results to {args.output_path}")
    except Exception as e:
        logger.error(f"Failed to write results to {args.output_path}: {e}")
        sys.exit(1)

    # Print summary to console
    logger.info("Evaluation completed. Summary:")
    if "results" in results:
        for task_name, metrics in results["results"].items():
            logger.info(f"\nTask: {task_name}")
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {metric_name}: {value:.4f}")

    if "groups" in results:
        logger.info("\nGroup-level averages:")
        for grp, grp_metrics in results["groups"].items():
            for metric_name, value in grp_metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {grp} {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
