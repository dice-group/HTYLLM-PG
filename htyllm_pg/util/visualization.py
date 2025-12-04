import matplotlib.pyplot as plt
import io
import numpy as np
from PIL import Image

def create_expert_heatmap(expert_counts):
    """
    Creates a heatmap of expert usage across layers.
    expert_counts: dict of {layer_name: tensor_counts}
    """
    # Filter None and sort layers by index (assuming format "layer_X")
    valid_layers = {k: v for k, v in expert_counts.items() if v is not None}
    if not valid_layers:
        return None
        
    sorted_layers = sorted(valid_layers.items(), key=lambda x: int(x[0].split('_')[1]))
    
    all_counts = []
    layer_boundaries = [0]
    layer_names = []
    
    for name, counts in sorted_layers:
        # counts is a tensor
        c = counts.detach().cpu().float().numpy()
        # Normalize to percentage for better visualization
        total = c.sum()
        if total > 0:
            c = (c / total) * 100
        
        all_counts.append(c)
        layer_boundaries.append(layer_boundaries[-1] + len(c))
        layer_names.append(name)
        
    if not all_counts:
        return None

    data = np.concatenate(all_counts)
    data = data.reshape(1, -1) # 1 x TotalExperts
    
    # Create figure
    # Width depends on number of experts, but keep it reasonable
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(data, cmap='viridis', aspect='auto', vmin=0, vmax=100)
    
    # Add vertical lines to separate layers
    for b in layer_boundaries[1:-1]: # Don't draw on edges
        ax.axvline(x=b - 0.5, color='white', linewidth=2)
        
    # Add layer labels
    centers = [(layer_boundaries[i] + layer_boundaries[i+1])/2 - 0.5 for i in range(len(layer_boundaries)-1)]
    ax.set_xticks(centers)
    ax.set_xticklabels(layer_names, rotation=45)
    ax.set_yticks([])
    ax.set_title("Expert Usage (%) per Layer")
    
    # Add colorbar
    cbar = plt.colorbar(im, orientation='horizontal', pad=0.2)
    cbar.set_label('% Usage')
    
    plt.tight_layout()
    
    # Save to buffer and convert to PIL Image for wandb compatibility
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)
