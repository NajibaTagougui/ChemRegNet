"""
Gradient-Based Attention Visualization (Model-Agnostic)
Works with GCN, GAT, or any PyTorch model

Run: python gradient_attention.py
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.cm as cm

def compute_gradient_attention(model, data, target_value=None, device='cpu'):
    """
    Compute attention via input gradients (saliency maps)
    Works with any differentiable model
    """
    model.eval()
    data = data.to(device)
    data.x.requires_grad_(True)
    
    # Forward pass
    output = model(data)
    
    # Use either target value or maximize output
    if target_value is not None:
        loss = F.mse_loss(output, torch.tensor([[target_value]]).to(device))
    else:
        loss = output.sum()
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Get gradients (attention proxy)
    gradients = data.x.grad.detach().cpu().numpy()
    
    # Aggregate gradients across features (L2 norm)
    atom_attention = np.linalg.norm(gradients, axis=1)
    
    # Normalize
    if atom_attention.max() > 0:
        atom_attention = atom_attention / atom_attention.max()
    
    return atom_attention


def visualize_saliency(smiles, atom_attention, save_path=None):
    """Visualize saliency-based attention"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Get coordinates
    Chem.rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    positions = [(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y) 
                for i in range(mol.GetNumAtoms())]
    
    # Molecular structure
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        x1, y1 = positions[i]
        x2, y2 = positions[j]
        axes[0].plot([x1, x2], [y1, y2], 'k-', linewidth=2)
    
    # Atoms colored by attention
    cmap = cm.get_cmap('hot')
    for i, (x, y) in enumerate(positions):
        color = cmap(atom_attention[i])
        axes[0].scatter(x, y, s=400 * (0.3 + atom_attention[i]), 
                       c=[color], edgecolors='black', linewidth=1.5)
        axes[0].text(x, y, mol.GetAtomWithIdx(i).GetSymbol(), 
                    ha='center', va='center', fontsize=10, fontweight='bold')
    
    axes[0].set_title('Gradient-Based Attention (Saliency)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Bar chart
    atoms = [f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}" for i in range(mol.GetNumAtoms())]
    axes[1].bar(range(len(atoms)), atom_attention, color='coral', alpha=0.7)
    axes[1].set_xticks(range(len(atoms)))
    axes[1].set_xticklabels(atoms, rotation=45, ha='right')
    axes[1].set_ylabel('Attention Score', fontsize=12)
    axes[1].set_title('Per-Atom Saliency', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Saliency Map: {Chem.MolToSmiles(mol)}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()