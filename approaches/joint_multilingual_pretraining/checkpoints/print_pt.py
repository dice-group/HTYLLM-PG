def print_model_structure(d, indent=0):
    """
    Recursively prints the structure of a nested dictionary,
    displaying tensor shapes and dtypes.
    """
    # Use d.items() if it's a dict, otherwise, handle other types
    if not isinstance(d, dict):
        print('  ' * indent + str(d))
        return

    for key, value in d.items():
        # Print the key with indentation
        print('  ' * indent + str(key), end=': ')

        if isinstance(value, dict):
            # If the value is another dictionary, recurse
            print() # Print a newline before diving deeper
            print_model_structure(value, indent + 1)
        elif isinstance(value, torch.Tensor):
            # If the value is a tensor, print its details
            print(f"Tensor(shape={value.shape}, dtype={value.dtype})")
        else:
            # For other types, print their type and a snippet of their value
            val_str = str(value)
            if len(val_str) > 80:
                val_str = val_str[:80] + '...'
            print(f"{type(value).__name__} = {val_str}")

import torch

# Replace with the actual path to your checkpoint file
checkpoint_path = 'checkpoints/moe_pretrain_mcore/iter_0002500/mp_rank_00/model_optim_rng.pt' 

print(f"Loading checkpoint from: {checkpoint_path}")

# Load the checkpoint onto the CPU
# WARNING: Only do this with files you trust.
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("Checkpoint loaded successfully!")

# Now, use this function to inspect the 'model' part of the checkpoint
if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
    print("\n" + "="*50)
    print("           Model State Dictionary Structure")
    print("="*50)
    print_model_structure(checkpoint['model'])
else:
    print("\nCould not find a 'model' dictionary in the checkpoint.")

# You can also inspect the 'args' object
if 'args' in checkpoint:
    print("\n" + "="*50)
    print("             Training Arguments (args)")
    print("="*50)
    # The 'args' object is not a dict, so we can't use our function directly.
    # We can print its attributes.
    args = checkpoint['args']
    for arg, value in sorted(vars(args).items()):
         print(f"- {arg}: {value}")
