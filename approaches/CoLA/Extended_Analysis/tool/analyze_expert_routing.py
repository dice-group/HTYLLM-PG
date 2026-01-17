#!/usr/bin/env python3
"""
Expert Routing Analysis for CoLA/HydraLoRA Checkpoints.

This script runs inference on language-specific test sets and collects
expert routing statistics for analysis and visualization.

Usage:
    python analyze_expert_routing.py \
        --base_model meta-llama/Llama-3.1-8B \
        --adapter_checkpoint ./checkpoints/checkpoint-5000 \
        --adapter_type hydralora \
        --test_data ./data/language_test_sets \
        --output ./analysis/checkpoint-5000
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging first (needed before _ensure_local_peft)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _ensure_local_peft():
    """
    Load repo-local PEFT with CoLA/Hydra support, not site-packages.
    
    This is critical because:
    1. pip's peft package doesn't have CoLA/HydraLoRA
    2. Python caches modules, so if peft was imported before, it won't reload
    3. We must force-reload from our local LLaMA-Factory/src/peft
    """
    # Clear any cached peft modules to force reload
    for name in list(sys.modules.keys()):
        if name == "peft" or name.startswith("peft."):
            del sys.modules[name]
    
    # Path: Extended_Analysis/tool/ → Extended_Analysis/ → CoLA/ → LLaMA-Factory/src
    repo_root = Path(__file__).resolve().parent.parent.parent  # CoLA/
    peft_path = repo_root / "LLaMA-Factory" / "src"
    
    if not (peft_path / "peft" / "__init__.py").exists():
        raise FileNotFoundError(
            f"Local PEFT not found at {peft_path}/peft. "
            "Ensure LLaMA-Factory submodule is cloned."
        )
    
    if str(peft_path) not in sys.path:
        sys.path.insert(0, str(peft_path))
    
    # Force-load local peft module using importlib
    import importlib.util
    peft_init = peft_path / "peft" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "peft", peft_init, 
        submodule_search_locations=[str(peft_path / "peft")]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load local peft from {peft_init}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules["peft"] = module
    spec.loader.exec_module(module)
    
    logger.info(f"Loaded local PEFT from: {peft_path}")
    return module


# Load local PEFT with CoLA/HydraLoRA support
_peft = _ensure_local_peft()
PeftModel = _peft.PeftModel


class LanguageDataset(Dataset):
    """Dataset for loading language-specific JSONL files."""
    
    def __init__(self, jsonl_file: Path, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        logger.info(f"Loading dataset from {jsonl_file}")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)
                self.examples.append(example)
        
        logger.info(f"Loaded {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Get text (handle different field names)
        text = example.get('text', example.get('content', ''))
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'original_text': text[:100]  # Keep snippet for debugging
        }


class RoutingCollector:
    """
    Collects expert routing statistics during forward passes.
    
    This class attaches to CoLA/HydraLoRA layers via forward hooks
    to capture routing decisions.
    """
    
    def __init__(self, adapter_type: str = 'cola'):
        self.adapter_type = adapter_type
        self.routing_data = defaultdict(lambda: defaultdict(list))
        self.current_language = None
        self.hooks = []
    
    def attach_hooks(self, model):
        """Attach forward hooks to adapter layers."""
        logger.info(f"Attaching routing collectors for {self.adapter_type}")
        
        hook_count = 0
        for name, module in model.named_modules():
            # Check module's class hierarchy for CoLA or HydraLoRA
            module_classes = [cls.__name__ for cls in module.__class__.__mro__]
            
            if self.adapter_type == 'cola':
                # CoLA uses ColaLayer base class, actual modules are Linear/Embedding/Conv2d
                if 'ColaLayer' in module_classes:
                    hook = module.register_forward_hook(
                        self._make_hook(name)
                    )
                    self.hooks.append(hook)
                    hook_count += 1
                    logger.debug(f"Attached hook to CoLA layer: {name} ({module.__class__.__name__})")
            
            elif self.adapter_type == 'hydralora':
                # HydraLoRA uses HydraLoraLayer (note: 'Lora' not 'LoRA')
                if 'HydraLoraLayer' in module_classes:
                    hook = module.register_forward_hook(
                        self._make_hook(name)
                    )
                    self.hooks.append(hook)
                    hook_count += 1
                    logger.debug(f"Attached hook to HydraLoRA layer: {name} ({module.__class__.__name__})")
        
        logger.info(f"Attached {hook_count} routing hooks")
        
        if hook_count == 0:
            logger.warning(
                f"No {self.adapter_type} layers found! " 
                "Check that the adapter type matches the checkpoint."
            )
    
    def _make_hook(self, layer_name: str):
        """Create a forward hook for a specific layer."""
        
        def hook(module, input, output):
            """Forward hook to capture routing decisions."""
            try:
                # CoLA stores routing data in _caches dict during forward pass
                # Check for cached routing state (set by _cache_router_state)
                router_logits = None
                if hasattr(module, '_caches') and isinstance(module._caches, dict):
                    router_logits = module._caches.get('cola_router_logits', None)
                
                # If no cache, try legacy attribute names
                if router_logits is None:
                    if hasattr(module, '_last_routing_probs'):
                        router_probs = module._last_routing_probs
                    elif hasattr(module, 'router_probs'):
                        router_probs = module.router_probs
                    else:
                        # No routing info available
                        return
                    
                    if hasattr(module, '_last_expert_indices'):
                        expert_indices = module._last_expert_indices
                    elif hasattr(module, 'expert_indices'):
                        expert_indices = module.expert_indices
                    else:
                        expert_indices = torch.argmax(router_probs, dim=-1)
                else:
                    # Extract from cache - logits are raw router outputs
                    # Compute probabilities and top-k indices
                    router_probs = torch.softmax(router_logits.to(torch.float32), dim=-1)
                    
                    # Get top-k indices (CoLA uses top_k, default 1)
                    top_k = getattr(module, 'top_k', 1)
                    topv, topi = torch.topk(router_logits, top_k, dim=-1)
                    expert_indices = topi[..., 0]  # Take first expert for statistics
                
                # Store routing decision
                if self.current_language is not None:
                    self.routing_data[self.current_language][layer_name].append({
                        'expert_indices': expert_indices.detach().cpu(),
                        'router_probs': router_probs.detach().cpu(),
                        'num_tokens': expert_indices.numel()
                    })
            
            except Exception as e:
                # Log errors for debugging but don't break forward pass
                logger.debug(f"Hook error in {layer_name}: {e}")
                pass
        
        return hook
    
    def remove_hooks(self):
        """Remove all attached hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def set_language(self, language_id: str):
        """Set current language for routing collection."""
        self.current_language = language_id
    
    def get_language_stats(self, language_id: str) -> Dict:
        """Get aggregated statistics for a language."""
        stats = self.routing_data[language_id]
        
        # Aggregate counts per layer per expert
        aggregated = {}
        for layer_name, records in stats.items():
            expert_counts = defaultdict(int)
            total_tokens = 0
            
            for record in records:
                # Count tokens routed to each expert
                indices = record['expert_indices'].flatten()
                for idx in indices:
                    expert_counts[int(idx)] += 1
                total_tokens += record['num_tokens']
            
            aggregated[layer_name] = {
                'expert_counts': dict(expert_counts),
                'total_tokens': total_tokens
            }
        
        return aggregated
    
    def save_raw_data(self, output_dir: Path, language_id: str):
        """Save raw routing data for a language."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = self.get_language_stats(language_id)
        output_file = output_dir / f"{language_id}_routing.json"
        
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved raw routing data for '{language_id}' to {output_file}")


def load_model_and_adapter(
    base_model: str,
    adapter_checkpoint: str,
    device: str = 'cuda'
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load base model with adapter checkpoint."""
    logger.info(f"Loading base model: {base_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    logger.info(f"Loading adapter: {adapter_checkpoint}")
    model = PeftModel.from_pretrained(
        base_model_obj,
        adapter_checkpoint
    )
    model.eval()
    
    logger.info("Model loaded successfully")
    return model, tokenizer


def analyze_language(
    model,
    tokenizer,
    language_file: Path,
    language_id: str,
    collector: RoutingCollector,
    batch_size: int = 16,
    max_sequences: Optional[int] = None
) -> Dict:
    """Run inference on language-specific dataset and collect routing stats."""
    logger.info(f"Analyzing language: {language_id}")
    
    # Set current language in collector
    collector.set_language(language_id)
    
    # Load dataset
    dataset = LanguageDataset(language_file, tokenizer)
    
    if max_sequences is not None:
        dataset.examples = dataset.examples[:max_sequences]
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Run inference
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Processing {language_id}"):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            
            # Forward pass (routing gets collected via hooks)
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Get aggregated statistics
    stats = collector.get_language_stats(language_id)
    
    logger.info(f"Completed analysis for '{language_id}': {len(stats)} layers")
    return stats


def create_routing_matrix(
    all_language_stats: Dict[str, Dict],
    languages: List[str],
    num_layers: int,
    num_experts: int
) -> np.ndarray:
    """
    Create routing matrix from collected statistics.
    
    Returns:
        routing_matrix: [num_languages, num_layers, num_experts]
    """
    logger.info("Creating routing matrix")
    
    routing_matrix = np.zeros(
        (len(languages), num_layers, num_experts),
        dtype=np.int64
    )
    
    for lang_idx, lang in enumerate(languages):
        if lang not in all_language_stats:
            logger.warning(f"No stats found for language '{lang}'")
            continue
        
        stats = all_language_stats[lang]
        
        for layer_name, layer_stats in stats.items():
            # Extract layer index from name (e.g., "base_model.model.model.layers.15.self_attn.q_proj" -> 15)
            # Look for "layers.X" pattern in the full path
            try:
                parts = layer_name.split('.')
                layer_idx = None
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts):
                        layer_idx = int(parts[i + 1])
                        break
                
                if layer_idx is None:
                    logger.debug(f"Could not extract layer index from: {layer_name}")
                    continue
                    
                if layer_idx >= num_layers:
                    logger.debug(f"Layer index {layer_idx} >= num_layers {num_layers}, skipping")
                    continue
            except (ValueError, IndexError) as e:
                logger.debug(f"Error parsing layer name '{layer_name}': {e}")
                continue
            
            # Fill in expert counts
            expert_counts = layer_stats['expert_counts']
            for expert_idx, count in expert_counts.items():
                if expert_idx < num_experts:
                    routing_matrix[lang_idx, layer_idx, expert_idx] = count
    
    return routing_matrix


def main():
    parser = argparse.ArgumentParser(description="Analyze expert routing for CoLA/HydraLoRA checkpoints")
    parser.add_argument('--base_model',type=str,required=True,help='Base model name or path (e.g., meta-llama/Llama-2-7b-hf)')
    parser.add_argument('--adapter_checkpoint', type=str, required=True, help='Path to adapter checkpoint directory')
    parser.add_argument('--adapter_type', type=str, choices=['cola', 'hydralora'], default=None, help='Type of adapter (cola or hydralora). Auto-detected from adapter_config.json if not specified.')
    parser.add_argument('--test_data', type=Path, required=True, help='Directory containing language-specific JSONL files')
    parser.add_argument('--languages', type=str, default=None, help='Comma-separated list of languages to analyze (default: all in test_data)')
    parser.add_argument('--output', type=Path, required=True, help='Output directory for routing statistics')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference (default: 16)')
    parser.add_argument('--max_sequences', type=int, default=None, help='Maximum sequences per language (default: all available)')
    parser.add_argument('--num_layers', type=int, default=None, help='Number of layers in model. Auto-detected from base model config if not specified.')
    parser.add_argument('--num_experts', type=int, default=None, help='Number of experts per layer. Auto-detected from adapter_config.json if not specified.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Auto-detect from adapter_config.json if not specified
    adapter_config_path = Path(args.adapter_checkpoint) / "adapter_config.json"
    if adapter_config_path.exists():
        logger.info(f"Reading adapter config from {adapter_config_path}")
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        # Auto-detect adapter_type
        if args.adapter_type is None:
            peft_type = adapter_config.get('peft_type', '').lower()
            if peft_type in ['cola', 'hydralora']:
                args.adapter_type = peft_type
                logger.info(f"Auto-detected adapter_type: {args.adapter_type}")
            else:
                logger.error(f"Could not auto-detect adapter_type from peft_type: {peft_type}")
                raise ValueError("Please specify --adapter_type (cola or hydralora)")
        
        # Auto-detect num_experts (CoLA uses cola_num_experts, HydraLoRA uses num_experts)
        if args.num_experts is None:
            # Try adapter-specific field names first
            num_experts = adapter_config.get('cola_num_experts')
            if num_experts is None:
                num_experts = adapter_config.get('num_experts')
            if num_experts is None:
                num_experts = 4  # Default fallback
            args.num_experts = num_experts
            logger.info(f"Auto-detected num_experts: {args.num_experts}")
    else:
        logger.warning(f"adapter_config.json not found at {adapter_config_path}")
        if args.adapter_type is None:
            raise ValueError("adapter_config.json not found. Please specify --adapter_type")
        if args.num_experts is None:
            args.num_experts = 4
            logger.warning(f"Using default num_experts: {args.num_experts}")
    
    # Auto-detect num_layers from base model if not specified
    if args.num_layers is None:
        # Will be detected after model loading
        args.num_layers = None  # Placeholder, will be set after model load
    
    # Get list of languages
    if args.languages:
        languages = [lang.strip() for lang in args.languages.split(',')]
    else:
        # Auto-detect from test data directory
        languages = [
            f.stem for f in args.test_data.glob('*.jsonl')
        ]
    
    logger.info(f"Analyzing {len(languages)} languages: {languages}")
    
    # Load model and adapter
    model, tokenizer = load_model_and_adapter(
        args.base_model,
        args.adapter_checkpoint,
        args.device
    )
    
    # Auto-detect num_layers from model config if not specified
    if args.num_layers is None:
        if hasattr(model.config, 'num_hidden_layers'):
            args.num_layers = model.config.num_hidden_layers
            logger.info(f"Auto-detected num_layers: {args.num_layers}")
        else:
            args.num_layers = 32  # Default fallback
            logger.warning(f"Could not auto-detect num_layers, using default: {args.num_layers}")
    
    # Create routing collector
    collector = RoutingCollector(adapter_type=args.adapter_type)
    collector.attach_hooks(model)
    
    # Analyze each language
    all_language_stats = {}
    raw_data_dir = args.output / 'raw_stats'
    
    for lang in languages:
        lang_file = args.test_data / f"{lang}.jsonl"
        
        if not lang_file.exists():
            logger.warning(f"File not found: {lang_file}, skipping")
            continue
        
        stats = analyze_language(
            model,
            tokenizer,
            lang_file,
            lang,
            collector,
            args.batch_size,
            args.max_sequences
        )
        
        all_language_stats[lang] = stats
        collector.save_raw_data(raw_data_dir, lang)
    
    # Create routing matrix
    routing_matrix = create_routing_matrix(
        all_language_stats,
        languages,
        args.num_layers,
        args.num_experts
    )
    
    # Save routing matrix
    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / 'routing_matrix.npz'
    
    np.savez(
        output_file,
        routing_matrix=routing_matrix,
        languages=languages,
        num_layers=args.num_layers,
        num_experts=args.num_experts
    )
    
    logger.info(f"Saved routing matrix to {output_file}")
    
    # Save metadata
    metadata = {
        'base_model': args.base_model,
        'adapter_checkpoint': args.adapter_checkpoint,
        'adapter_type': args.adapter_type,
        'num_languages': len(languages),
        'languages': languages,
        'num_layers': args.num_layers,
        'num_experts': args.num_experts,
        'routing_matrix_shape': list(routing_matrix.shape)
    }
    
    metadata_file = args.output / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata to {metadata_file}")
    logger.info("Analysis complete!")
    
    # Cleanup
    collector.remove_hooks()


if __name__ == '__main__':
    main()
