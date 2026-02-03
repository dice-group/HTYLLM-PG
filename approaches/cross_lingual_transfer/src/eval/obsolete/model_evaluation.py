import os
import glob
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from transformers import BertForPreTraining, BertTokenizer
import pandas as pd
from pathlib import Path

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

class BinaryPretrainingDataset(Dataset):
    def __init__(self, bin_file: str, block_size: int):
        self.data = np.memmap(bin_file, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, idx):
        i = idx * self.block_size
        input_ids = torch.tensor(self.data[i: i + self.block_size], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        rand = torch.rand(input_ids.shape)
        mask_arr = (rand < 0.15) & (input_ids != tokenizer.cls_token_id) & (input_ids != tokenizer.sep_token_id)
        input_ids[mask_arr] = tokenizer.mask_token_id

        next_sentence_label = torch.tensor(1, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "next_sentence_label": next_sentence_label
        }

def compute_loss_on_batch(model, batch):
    with torch.no_grad():
        outputs = model(**batch)
        return outputs.loss.item()

def evaluate_on_gpu(rank, world_size, k, block_size, tokenized_data, result_list):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = BertForPreTraining.from_pretrained("./results/mbert_fine_tuned_model", local_files_only=True)
    model.to(rank)
    model = DDP(model, device_ids=[rank])
    model.eval()

    for bin_file in glob.glob(os.path.join(tokenized_data, "*.bin")):
        lang = Path(bin_file).stem
        full_dataset = BinaryPretrainingDataset(bin_file, block_size)

        num_samples = min(k, len(full_dataset))
        if num_samples == 0:
            if rank == 0:
                print(f"Skipping {lang} due to empty dataset.")
            continue

        indices = np.linspace(0, len(full_dataset) - 1, num_samples, dtype=int)
        subset = Subset(full_dataset, indices)
        sampler = DistributedSampler(subset, num_replicas=world_size, rank=rank, shuffle=False)
        loader = DataLoader(subset, batch_size=8, sampler=sampler)

        local_losses = []
        for batch in loader:
            batch = {k: v.to(rank) for k, v in batch.items()}
            loss = compute_loss_on_batch(model, batch)
            local_losses.append(loss)

        local_avg_loss = torch.tensor(np.mean(local_losses), device=rank)
        dist.all_reduce(local_avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = local_avg_loss.item() / world_size

        if rank == 0:
            result_list.append({"language": lang, "avg_loss": avg_loss})
            print(f"{lang}: loss={avg_loss:.4f}")

    dist.destroy_process_group()

def main():
    world_size = torch.cuda.device_count()
    k = 1000
    block_size = 256
    tokenized_data = "tokenized_data"
    manager = mp.Manager()
    result_list = manager.list()  # Shared list for multiprocessing-safe results

    mp.spawn(
        evaluate_on_gpu,
        args=(world_size, k, block_size, tokenized_data, result_list),
        nprocs=world_size,
        join=True
    )

    # Save results only once
    df = pd.DataFrame(list(result_list))
    df.to_csv("zeroshot_eval_results.csv", index=False)
    print("Saved evaluation results to zeroshot_eval_results.csv")

if __name__ == "__main__":
    main()
