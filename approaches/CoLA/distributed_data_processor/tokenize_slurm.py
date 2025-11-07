import argparse
import gzip
import os
import tempfile

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional
from datasets import Dataset
from transformers import AutoTokenizer


@dataclass(frozen=True)
class TextChunk:
    path: str
    start: int = 0
    end: Optional[int] = None


def _resolve_rank_world(rank: Optional[int], world_size: Optional[int]) -> tuple[int, int]:
    # world rank = size of all tasks. Priority order: CLI flag, SLURM envs, fallback to zero
    if rank is not None:
        resolved_rank = rank
    else:
        env_rank = (
            os.getenv("SLURM_PROCID")
            or os.getenv("RANK")
            or os.getenv("LOCAL_RANK")
            or os.getenv("SLURM_ARRAY_TASK_ID")
            or 0
        )
        resolved_rank = int(env_rank)

    if world_size is not None:
        resolved_world = world_size
    else:
        env_world = (
            os.getenv("SLURM_NTASKS")
            or os.getenv("WORLD_SIZE")
            or os.getenv("SLURM_ARRAY_TASK_COUNT")
            or 1
        )
        resolved_world = int(env_world)

    return resolved_rank, max(1, resolved_world)


def read_folder_chunks(folder_path: str, max_chunk_bytes: Optional[int]) -> List[TextChunk]:
    # Walk the tree of language dirs and emit chunk metadata per file
    chunk_bytes = max_chunk_bytes if max_chunk_bytes and max_chunk_bytes > 0 else None
    chunks: List[TextChunk] = []
    for root, _, files in os.walk(folder_path):
        for file in sorted(files):
            if not file.endswith(".gz") and not file.endswith(".jsonl"):
                continue
            path = os.path.join(root, file)
            chunks.extend(_chunk_file(path, chunk_bytes))
    return sorted(chunks, key=lambda chunk: (chunk.path, chunk.start))


def _chunk_file(path: str, max_chunk_bytes: Optional[int]) -> List[TextChunk]:
    compressed = path.endswith(".gz")
    if compressed or not max_chunk_bytes:
        # gzip: single chunk because random access is expensive.
        return [TextChunk(path=path)]

    size = os.path.getsize(path)
    if size <= max_chunk_bytes:
        return [TextChunk(path=path)]

    result: List[TextChunk] = []
    with open(path, "rb") as handle:
        start = 0
        while start < size:
            target = min(start + max_chunk_bytes, size)
            if target < size:
                handle.seek(target)
                _ = handle.readline()
                new_end = handle.tell()
                if new_end <= start:
                    new_end = size
            else:
                new_end = size
            result.append(TextChunk(path=path, start=start, end=new_end))
            start = new_end
    return result


def _assign_chunks(chunks: List[TextChunk], rank: int, world_size: int) -> List[TextChunk]:
    # Round-robin assignment keeps work balanced across ranks.
    if world_size <= 1:
        return chunks
    return [chunk for idx, chunk in enumerate(chunks) if idx % world_size == rank]


def iter_chunk_lines(chunks: Iterable[TextChunk]) -> Iterator[dict]:
    # Stream JSONL records from whatever files belong to this rank.
    for chunk in chunks:
        try:
            yield from _iter_compressed_lines(chunk.path)
        except Exception as exc:
            print(f"Error reading {chunk.path}: {exc}")


def _iter_compressed_lines(path: str) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line and len(line) > 10:
                yield {"text": line}

def tokenize_fn(batch, tokenizer):
    #tokenized = tokenizer(batch["text"], truncation=True, padding=True)
    tokenized = tokenizer(batch["text"], truncation=True, padding=True, max_length=1024)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def main(args):
    print(f"tokenizer used for tokenization: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    rank, world_size = _resolve_rank_world(args.rank, args.world_size)
    print(f"Read data from: {args.data_dir} (rank {rank}/{world_size})")

    all_chunks = read_folder_chunks(args.data_dir, args.max_chunk_bytes)            # We make a list of all file chunks
    assigned_chunks = _assign_chunks(all_chunks, rank, world_size)                  # We only keep the ones this process should handle
    if not assigned_chunks:
        raise RuntimeError(f"No chunks assigned to rank {rank}. Check input directory or chunk filters.")

    # Use a temporary folder so the dataset cache is only kept for this run
    with tempfile.TemporaryDirectory(prefix="tok_simple_") as cache_dir:
        raw_dataset = Dataset.from_generator(
            lambda: iter_chunk_lines(assigned_chunks),
            cache_dir=cache_dir,
        )

        print("Start tokenizing")
        # Tokenize the data chunks; use num_proc to run in parallel within this job.
        tokenized_dataset = raw_dataset.map(
            lambda batch: tokenize_fn(batch, tokenizer),
            batched=True,
            remove_columns=["text"],
            num_proc=args.num_proc
        )

        output_dir = args.save_tokenized_data_dir
        if world_size > 1:
            output_dir = os.path.join(output_dir, f"rank_{rank:05d}")
        # Save this ranks part of the data; it can be combined later.
        tokenized_dataset.save_to_disk(output_dir)
        print(f"tokenized data saved to here: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .gz files with text data. e.g sampled Fineweb data")
    parser.add_argument("--save_tokenized_data_dir", type=str, required=True, help="Directory to save the tokenized dataset.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Pretrained tokenizer model name. You can also use own trained tokenizers paths here")
    parser.add_argument("--num_proc", type=int, default=1, help="Number of processes for tokenization.")
    parser.add_argument("--max_chunk_bytes", type=int, default=None, help="Split large uncompressed files into chunks of this size (bytes).")
    parser.add_argument("--rank",type=int, default=None, help="Process rank override (defaults to SLURM_PROCID/RANK/LOCAL_RANK or SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--world_size", type=int, default=None, help="Total ranks override (defaults to SLURM_NTASKS/WORLD_SIZE or SLURM_ARRAY_TASK_COUNT).",)
    args = parser.parse_args()
    main(args)
