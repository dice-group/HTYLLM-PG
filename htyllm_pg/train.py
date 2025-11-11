from numpy import argmax
from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset
import deepspeed
from deepspeed import comm
import argparse
from htyllm_pg.model_builder import moe_builder
from tqdm.auto import tqdm

from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer


def get_args() -> argparse.Namespace :
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", default=8, type=int, help="Number of workers for the Dataloaders!")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs of the training data!")
    parser.add_argument("--batch-size", default=8, type=int, dest="batch_size", help="Batch size for training and testing!")
    parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate for AdamW optimizer!")
    parser.add_argument("--weight-decay", default=1e-4, type=float, dest="weight_decay")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab_size = 32_000
    seq_len = 1_000
    args = get_args()

    model_pytorch = moe_builder(vocab_size=vocab_size, max_seq_len=seq_len)

    base_params = {
        "params": [p for p in model_pytorch.parameters() if p.requires_grad],
        "name": "parameters",
    }

    # 2) let DeepSpeed split into MoE / non-MoE param groups
    param_groups = split_params_into_different_moe_groups_for_optimizer(base_params)

    # 3) construct AdamW on those groups (some will get `moe: True` internally)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    model, optimizer, _, _ = deepspeed.initialize(
        model = model_pytorch,
        optimizer=optimizer,
        args=args,
    )

    RANK = comm.get_rank()

    criterion = nn.CrossEntropyLoss().to(device)
    

    train_dataset = DummyTextDataset(vocab_size=vocab_size, seq_len=seq_len, num_samples=8_000)
    test_dataset = DummyTextDataset(vocab_size=vocab_size, seq_len=seq_len, num_samples=2_000)

    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=args.workers, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=args.workers, batch_size=args.batch_size)

    for epoch in range(args.epochs): #normalerweise 1

        model.train()
        for step, (input_ids, target) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            input_ids = input_ids.to(device)
            target = target.to(device)

            output, l_aux = model(input_ids)
            
            ce_loss = criterion(output.float().transpose(1, 2), target)

            loss = ce_loss + 0.01 * l_aux

            model.backward(loss)
            model.step()
            
            if RANK == 0 and step % 100 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{step}], Train Loss: {loss.item():.4f}")

    # Evaluation after training is complete
    if RANK == 0:
        print(f"\n{'='*50}")
        print("Training complete! Starting evaluation...")
        print(f"{'='*50}\n")
    
    model.eval()
    with torch.inference_mode():
        test_loss_sum = 0
        num_test_batches = 0
        
        for input_ids, target in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            target = target.to(device)

            output, l_aux = model(input_ids)
            test_loss = criterion(output.float().transpose(1,2), target) + 0.01 * l_aux
            test_loss_sum += test_loss.item()
            num_test_batches += 1
            
        avg_test_loss = test_loss_sum / num_test_batches
        
        if RANK == 0:
            print(f"\n{'='*50}")
            print(f"Final Test Loss: {avg_test_loss:.4f}")
            print(f"{'='*50}\n")
        
            # Test prediction 
            test_pred, _ = model(torch.arange(10).unsqueeze(0).to(device))
            print(f"Test prediction shape: {test_pred.shape}")
            print(f"Prediction for [0,...,9]: {torch.argmax(test_pred.squeeze()[9])}")
            print(f"{'='*50}\n")


class DummyTextDataset(Dataset):
    def __init__(self, vocab_size, seq_len, num_samples=10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.arange(self.seq_len + 1)
        inputs = x[:-1]
        targets = x[1:]

        return inputs, targets


if __name__ == "__main__":
    main()