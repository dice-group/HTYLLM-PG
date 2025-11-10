from numpy import argmax
from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset
import deepspeed
from deepspeed import comm
import argparse
from htyllm_pg.model_builder import moe_builder
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

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

    # Lists to track losses for plotting
    train_losses = []
    test_losses = []
    train_steps = []
    test_steps = []

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
            
            # Track training loss
            train_losses.append(loss.item())
            train_steps.append(step)
            
            if step % 100 == 0 and step != 0:
                model.eval()
                with torch.inference_mode():
                    test_loss_sum = 0
                    num_test_batches = 5  
                    
                    for i, (input_ids, target) in enumerate(test_dataloader):
                        if i >= num_test_batches:
                            break
                            
                        input_ids = input_ids.to(device)
                        target = target.to(device)

                        output, l_aux = model(input_ids)
                        test_loss = criterion(output.float().transpose(1,2), target) + 0.01 * l_aux
                        test_loss_sum += test_loss.item()
                        
                    avg_test_loss = test_loss_sum / num_test_batches
                    
                    # Track test loss
                    test_losses.append(avg_test_loss)
                    test_steps.append(step)
                    
                    if RANK == 0:
                        print(f"Rank: {RANK}")
                        print(f"{'='*30}\nEpoch [{epoch+1}/{args.epochs}], Step [{step}], "
                            f"Train Loss: {loss.item():.4f}\n"
                            f"Test Loss: {avg_test_loss:.4f}\n{'='*30}")
                    
                        # Test prediction 
                        test_pred, _ = model(torch.arange(10).unsqueeze(0).to(device))
                        print(test_pred.shape)
                        print("Prediction for [0,...,9]:", torch.argmax(test_pred.squeeze()[9]))
                    
                model.train()

    # Plot train and test loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_losses, label='Train Loss', alpha=0.7, linewidth=1)
    plt.plot(test_steps, test_losses, label='Test Loss', marker='o', linewidth=2)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss_plot_sign.png', dpi=150)
    print(f"\nLoss plot saved as 'loss_plot.png'")
    plt.show()


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