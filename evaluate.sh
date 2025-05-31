#!/bin/bash

# Simple evaluation script - runs lm-harness on the latest checkpoint
python evaluate_checkpoint.py --checkpoint_path checkpoints/pretrain_run_vm/checkpoint-157000 --batch_size 64 --limit 10 