# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""CALM training script for finetuning.

Hugging Face Trainer is used to train the CALM model.
Reference:
https://huggingface.co/docs/transformers/main_classes/trainer
"""

from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging
import datasets
import os
import torch
import random
import numpy as np
from glob import glob
from model import calm
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from transformers import TrainingArguments
from datasets import load_from_disk, DatasetDict

import torch
import math
from types import SimpleNamespace

def register_nan_hooks(model):
    """
    Registers forward‑ and backward‑hooks that raise as soon as a tensor
    containing NaN/Inf appears.  The exception message tells you which
    sub‑module broke.
    """
    def _check(name, where):
        def hook(module, inp, out):
            tensors = list(inp) + ([out] if not isinstance(out, tuple) else list(out))
            for t in tensors:
                if torch.is_tensor(t) and (not torch.isfinite(t).all()):
                    bad = t[~torch.isfinite(t)]
                    info = SimpleNamespace(module=name, place=where,
                                           shape=t.shape, min=float(bad.min()),
                                           max=float(bad.max()))
                    raise RuntimeError(f"{where}‑pass   NaN/Inf in {name}: {info}")
        return hook

    for name, m in model.named_modules():
        m.register_forward_hook(_check(name, "forward"))
        m.register_full_backward_hook(_check(name, "backward"))



_ANCHOR_MODEL_DIR = flags.DEFINE_string(
    'anchor_model_dir', None, 'anchor model path.'
)
_AUG_MODEL_DIR = flags.DEFINE_string('aug_model_dir', None, 'aug model path.')
_OUTPUT_DIR = flags.DEFINE_string('output_dir', None, 'output directory.')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 5e-6, 'learning rate.')
_EPOCHS = flags.DEFINE_integer('epochs', 3, 'number of epochs.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 1, 'batch size.')
_NUM_HEADS = flags.DEFINE_integer('num_heads', 1, 'number of heads.')
_NUM_CONNECTIONS = flags.DEFINE_integer(
    'num_connections', 2, 'number of connections.'
)
_CONNECTIONS = flags.DEFINE_list(
    'connections',
    None,
    'connections between the anchor and aug model. You cannot provide both'
    'connections and num_connections simultaneously.',
)
_EVAL_STEPS = flags.DEFINE_integer('eval_steps', 50, 'eval steps.')
_LOGGING_STEPS = flags.DEFINE_integer('logging_steps', 1, 'logging steps.')
_SAVE_STEPS = flags.DEFINE_integer('save_steps', 50, 'save steps.')
_MAX_STEPS = flags.DEFINE_integer('max_steps', 10000, 'max steps.')


def train(argv: Sequence[str]) -> None:
  
  def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

  set_seed(42)

  """Trains the CALM model."""
  del argv  # Unused.
  anchor_model_path = _ANCHOR_MODEL_DIR.value
  aug_model_path = _AUG_MODEL_DIR.value
  num_heads = _NUM_HEADS.value
  num_connections = _NUM_CONNECTIONS.value
  logging.info('anchor_model_path: %s', anchor_model_path)
  logging.info('aug_model_path: %s', aug_model_path)
  logging.info('Loading Tokenizer...')
  tokenizer = AutoTokenizer.from_pretrained(anchor_model_path)
  logging.info('Loading Composed Model...')
  calm_config = calm.CALMConfig(
      anchor_model=anchor_model_path,
      aug_model=aug_model_path,
      anchor_config=None,
      aug_config=None,
      num_connections=num_connections,
      num_heads=num_heads,
  )

  model = calm.CALM(calm_config)
  register_nan_hooks(model)
    
  data_root = "/data/fineweb2_subset_belebele"
  jsonl_files = [
    f for f in glob(os.path.join(data_root, "*/*.jsonl.gz"))
    if "sampling_stats" not in os.path.basename(f)
  ]

  # 80/20 train/test split
  split_index = int(0.8 * len(jsonl_files))
  train_files = jsonl_files[:split_index]
  test_files = jsonl_files[split_index:]

  # Load dataset using Hugging Face Datasets
  train_data = datasets.load_dataset(
      "json",
      data_files={"train": train_files, "test": test_files},
      split=None
  )
  
  def preprocess_function(examples):
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=512,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

  tokenized_path = "/data/finweb2_tokenized"
  columns_to_remove = ["text", "id", "metadata"]
  if os.path.exists(os.path.join(tokenized_path, "train")) and os.path.exists(os.path.join(tokenized_path, "test")):
    train_data = DatasetDict({
        "train": load_from_disk(os.path.join(tokenized_path, "train")),
        "test": load_from_disk(os.path.join(tokenized_path, "test"))
    })
  else:
    train_data = {
        split: data.map(preprocess_function, batched=True, num_proc=24).remove_columns(columns_to_remove)
        for split, data in train_data.items()
    }
    train_data["train"].save_to_disk(os.path.join(tokenized_path, "train"))
    train_data["test"].save_to_disk(os.path.join(tokenized_path, "test"))
    
  data_collator = DataCollatorForLanguageModeling(
      tokenizer=tokenizer, mlm=False
  )
  print("!!!!!!!!!!!!!!! EXample Train data", train_data["train"][0])

  epochs = _EPOCHS.value
  batch_size = _BATCH_SIZE.value
  learning_rate = _LEARNING_RATE.value
  output_dir = _OUTPUT_DIR.value
  eval_steps = _EVAL_STEPS.value
  logging_steps = _LOGGING_STEPS.value
  save_steps = _SAVE_STEPS.value
  max_steps = _MAX_STEPS.value
  training_args = TrainingArguments(
      output_dir=output_dir,
      overwrite_output_dir=True,
      num_train_epochs=epochs,
      do_train=True,
      do_eval=True,
      per_device_train_batch_size=batch_size,
      per_device_eval_batch_size=batch_size,
      eval_strategy='steps',  # pylint:disable=unexpected-keyword-arg
      eval_steps=eval_steps,
      logging_steps=logging_steps,
      save_steps=save_steps,
      max_steps=max_steps,
      learning_rate=learning_rate,
      max_grad_norm=1.0,
      #label_names=[],
      report_to=['tensorboard'],
      save_total_limit=2,
      #load_best_model_at_end=True,  
      #metric_for_best_model='eval_loss',
      #greater_is_better=False, 
      #save_strategy='steps',
      #logging_dir=os.path.join(output_dir, "logs"),
      #resume_from_checkpoint=True,
      #gradient_accumulation_steps=8,  
      #fp16=False,
  )

  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_data['train'],
      eval_dataset=train_data['test'],
      data_collator=data_collator,
      tokenizer=tokenizer,
  )
  
  batch = next(iter(trainer.get_train_dataloader()))
  with torch.autograd.detect_anomaly():
    loss = trainer.model(**batch).loss
  print(" \n \n \n !!!!!!!!!!!!!!!!!!!!!!!!!! first‑step loss =", loss.item())
  assert torch.isfinite(loss), "loss is NaN or Inf – still broken"

  trainer.can_return_loss = True

  trainer.train()

  trainer.save_model(
      output_dir,
  )

  print(f'Training complete! Model saved to {output_dir}')


if __name__ == '__main__':
  app.run(train)
