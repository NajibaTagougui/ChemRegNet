"""
ATTENTION VISUALIZATION FOR TRAINED MODEL (COMPLETELY FIXED)
Properly handles SMILES parsing and model compatibility

Run: python visualization.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Set seed
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class GCNModelWithGradients(nn.Module):
    def __init__(self, node_features_dim=20, hidden_dim=64, num_classes=1, dropout=0.2):
        super().__init__()
        
        self.conv1 = GCNConv(node_features_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.conv1(x, edge_index))
        x = self.batch_norm1(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x, edge_index))
        x = self.batch_norm2(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x, edge_index))
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        return self.predictor(x)


# ============================================================================
# IMPROVED MOLECULAR GRAPH CONSTRUCTION
# ============================================================================

def smiles_to_graph_with_coords(smiles):
    """Convert SMILES to graph with 2D coordinates - FIXED VERSION"""
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"    Could not parse SMILES: {smiles}")
            return None, None, None
        
        # Sanitize molecule
        try:
            Chem.SanitizeMol(mol)
        except:
            # If sanitization fails, try with sanitize=False
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return None, None, None
        
        # Generate 2D coordinates
        try:
            Chem.rdDepictor.Compute2DCoords(mol)
        except:
            # If coordinate generation fails, use a simple layout
            from rdkit.Chem import AllChem
            AllChem.Compute2DCoords(mol)
        
        # Get atom positions
        positions = []
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            positions.append((pos.x, pos.y))
        
        # Atom features
        node_features = []
        for atom in mol.GetAtoms():
            feat = [0.0] * 20
            
            atomic_num = atom.GetAtomicNum()
            if atomic_num == 6:
                feat[0] = 1.0
            elif atomic_num == 7:
                feat[1] = 1.0
            elif atomic_num == 8:
                feat[2] = 1.0
            elif atomic_num == 9:
                feat[3] = 1.0
            elif atomic_num == 16:
                feat[4] = 1.0
            elif atomic_num == 17:
                feat[5] = 1.0
            else:
                feat[6] = 1.0
            
            feat[7] = min(atom.GetDegree(), 5) / 5.0
            feat[8] = min(atom.GetTotalNumHs(), 4) / 4.0
            feat[9] = 1.0 if atom.GetIsAromatic() else 0.0
            feat[10] = 1.0 if atom.IsInRing() else 0.0
            feat[11] = atom.GetFormalCharge() / 3.0
            
            node_features.append(feat)
        
        # Edge indices
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i])
        
        if len(edge_indices) == 0:
            edge_indices = [[0, 0]]
        
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        
        data = Data(x=x, edge_index=edge_index)
        data.num_nodes = len(node_features)
        
        return data, positions, mol
        
    except Exception as e:
        print(f"    Error: {str(e)[:100]}")
        return None, None, None


# ============================================================================
# ATTENTION COMPUTATION METHODS
# ============================================================================

def compute_saliency_attention(model, data, device):
    """Compute via input gradients"""
    model.eval()
    data = data.to(device)
    
    data.x.requires_grad_(True)
    output = model(data)
    
    model.zero_grad()
    output.backward()
    
    if data.x.grad is not None:
        gradients = data.x.grad.detach().cpu().numpy()
        atom_saliency = np.linalg.norm(gradients, axis=1)
    else:
        atom_saliency = np.ones(data.num_nodes)
    
    if atom_saliency.max() > 0:
        atom_saliency = atom_saliency / atom_saliency.max()
    
    return atom_saliency


def compute_activation_attention(model, data, device):
    """Compute via activation magnitude"""
    model.eval()
    data = data.to(device)
    
    activations = []
    
    def hook_fn(module, input, output):
        activations.append(output.detach())
    
    handle = model.conv3.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = model(data)
    
    handle.remove()
    
    if activations:
        node_activations = activations[0].cpu().numpy()
        atom_attention = np.linalg.norm(node_activations, axis=1)
        if atom_attention.max() > 0:
            atom_attention = atom_attention / atom_attention.max()
        return atom_attention
    
    return np.ones(data.num_nodes)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_atom_attention(mol, positions, attention_scores, title, save_path=None, cmap_name='RdYlGn_r'):
    """Visualize attention scores on molecule atoms"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scores = np.array(attention_scores)
    if scores.max() > 0:
        scores_norm = scores / scores.max()
    else:
        scores_norm = scores
    
    cmap = cm.get_cmap(cmap_name)
    
    # Draw bonds
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i < len(positions) and j < len(positions):
            x1, y1 = positions[i]
            x2, y2 = positions[j]
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, zorder=1, alpha=0.6)
    
    # Draw atoms
    for i, (x, y) in enumerate(positions):
        if i < len(scores_norm):
            color = cmap(scores_norm[i])
            size = 800 * (0.3 + scores_norm[i] * 0.7)
        else:
            color = cmap(0)
            size = 400
        
        ax.scatter(x, y, s=size, c=[color], edgecolors='black', linewidth=1.5, zorder=2)
        if i < mol.GetNumAtoms():
            ax.text(x, y, mol.GetAtomWithIdx(i).GetSymbol(), 
                   ha='center', va='center', fontsize=11, fontweight='bold', zorder=3)
    
    if positions:
        ax.set_xlim(min(p[0] for p in positions) - 1, max(p[0] for p in positions) + 1)
        ax.set_ylim(min(p[1] for p in positions) - 1, max(p[1] for p in positions) + 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label('Attention Score', fontsize=12)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {save_path}")
    
    plt.close()
    return fig


def visualize_attention_barchart(mol, attention_scores, title, save_path=None):
    """Bar chart of per-atom attention scores"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    atoms = [f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}" for i in range(min(mol.GetNumAtoms(), len(attention_scores)))]
    scores = np.array(attention_scores[:len(atoms)])
    
    cmap = cm.get_cmap('RdYlGn_r')
    if scores.max() > 0:
        colors = [cmap(s / scores.max()) for s in scores]
    else:
        colors = [cmap(0) for _ in scores]
    
    bars = ax.bar(range(len(atoms)), scores, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(atoms)))
    ax.set_xticklabels(atoms, rotation=45, ha='right')
    ax.set_ylabel('Attention Score', fontsize=12)
    ax.set_xlabel('Atom', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {save_path}")
    
    plt.close()
    return fig


def visualize_comparative_attention(mol, positions, attention_methods, save_path=None):
    """Compare multiple attention methods"""
    n_methods = len(attention_methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
    
    if n_methods == 1:
        axes = [axes]
    
    for idx, (method_name, scores) in enumerate(attention_methods.items()):
        ax = axes[idx]
        
        scores_arr = np.array(scores)
        if scores_arr.max() > 0:
            scores_norm = scores_arr / scores_arr.max()
        else:
            scores_norm = scores_arr
        
        cmap = cm.get_cmap('RdYlGn_r')
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if i < len(positions) and j < len(positions):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, zorder=1, alpha=0.5)
        
        for i, (x, y) in enumerate(positions):
            if i < len(scores_norm):
                color = cmap(scores_norm[i])
                size = 700 * (0.3 + scores_norm[i] * 0.7)
                ax.scatter(x, y, s=size, c=[color], edgecolors='black', linewidth=1.5, zorder=2)
                if i < mol.GetNumAtoms():
                    ax.text(x, y, mol.GetAtomWithIdx(i).GetSymbol(), 
                           ha='center', va='center', fontsize=10, fontweight='bold', zorder=3)
        
        if positions:
            ax.set_xlim(min(p[0] for p in positions) - 1, max(p[0] for p in positions) + 1)
            ax.set_ylim(min(p[1] for p in positions) - 1, max(p[1] for p in positions) + 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(method_name, fontsize=12, fontweight='bold')
    
    plt.suptitle('Comparison of Attention Methods', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {save_path}")
    
    plt.close()


# ============================================================================
# TEST MOLECULES (SIMPLE ONES FIRST)
# ============================================================================

def get_test_molecules():
    """Simple test molecules that should work"""
    # Start with very simple molecules to test
    examples = {
        "Methane": "C",
        "Ethanol": "CCO",
        "Benzene": "C1=CC=CC=C1",
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    }
    return examples


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("ATTENTION VISUALIZATION FOR TRAINED MODEL")
    print("=" * 80)
    
    os.makedirs('attention_figures', exist_ok=True)
    
    # Load model
    print("\n[1/4] Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNModelWithGradients()
    
    model_path = 'best_model.pt'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"  Loaded model from {model_path}")
    else:
        print(f"  WARNING: {model_path} not found!")
        print("  Using untrained model - visualizations will show random patterns")
    
    model = model.to(device)
    model.eval()
    
    # Get molecules - start with simple ones
    print("\n[2/4] Loading molecules...")
    examples = get_test_molecules()
    print(f"  Loaded {len(examples)} molecules")
    
    # Process each molecule
    print("\n[3/4] Computing attention maps...")
    results = []
    
    for name, smiles in examples.items():
        print(f"\n  Processing: {name}")
        print(f"    SMILES: {smiles}")
        
        data, positions, mol = smiles_to_graph_with_coords(smiles)
        
        if data is None or positions is None or mol is None:
            print(f"    FAILED: Could not process {name}")
            continue
        
        print(f"    Atoms: {mol.GetNumAtoms()}, Bonds: {mol.GetNumBonds()}")
        
        # Create batch
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        
        # Compute attention
        print(f"    Computing saliency...")
        saliency = compute_saliency_attention(model, data, device)
        
        print(f"    Computing activation...")
        activation = compute_activation_attention(model, data, device)
        
        # Generate visualizations
        print(f"    Generating figures...")
        
        visualize_atom_attention(
            mol, positions, saliency,
            f'{name}: Saliency Map',
            f'attention_figures/{name.lower()}_saliency.png'
        )
        
        visualize_atom_attention(
            mol, positions, activation,
            f'{name}: Activation Attention',
            f'attention_figures/{name.lower()}_activation.png',
            cmap_name='plasma'
        )
        
        visualize_attention_barchart(
            mol, saliency,
            f'{name}: Per-Atom Saliency Scores',
            f'attention_figures/{name.lower()}_barchart.png'
        )
        
        attention_methods = {
            'Saliency': saliency,
            'Activation': activation,
        }
        
        visualize_comparative_attention(
            mol, positions, attention_methods,
            f'attention_figures/{name.lower()}_comparison.png'
        )
        
        results.append({
            'name': name,
            'smiles': smiles,
            'num_atoms': mol.GetNumAtoms(),
            'saliency': saliency.tolist(),
            'activation': activation.tolist(),
            'top_atoms': np.argsort(saliency)[-5:].tolist()
        })
        
        print(f"    SUCCESS! Figures saved for {name}")
    
    # Save results
    print("\n[4/4] Saving results...")
    
    with open('attention_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary grid
    if results:
        n_results = min(len(results), 6)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, result in enumerate(results[:n_results]):
            ax = axes[idx]
            smiles = result['smiles']
            saliency = np.array(result['saliency'])
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            Chem.rdDepictor.Compute2DCoords(mol)
            conf = mol.GetConformer()
            positions = [(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y) 
                        for i in range(mol.GetNumAtoms())]
            
            if saliency.max() > 0:
                saliency_norm = saliency / saliency.max()
            else:
                saliency_norm = saliency
            
            cmap = cm.get_cmap('RdYlGn_r')
            
            # Draw bonds
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                if i < len(positions) and j < len(positions):
                    x1, y1 = positions[i]
                    x2, y2 = positions[j]
                    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, zorder=1, alpha=0.6)
            
            # Draw atoms
            for i, (x, y) in enumerate(positions):
                if i < len(saliency_norm):
                    color = cmap(saliency_norm[i])
                    size = 500 * (0.3 + saliency_norm[i] * 0.7)
                    ax.scatter(x, y, s=size, c=[color], edgecolors='black', linewidth=1.5, zorder=2)
                    ax.text(x, y, mol.GetAtomWithIdx(i).GetSymbol(), 
                           ha='center', va='center', fontsize=9, fontweight='bold', zorder=3)
            
            if positions:
                ax.set_xlim(min(p[0] for p in positions) - 1, max(p[0] for p in positions) + 1)
                ax.set_ylim(min(p[1] for p in positions) - 1, max(p[1] for p in positions) + 1)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(f"{result['name']}", fontsize=10, fontweight='bold')
        
        for idx in range(n_results, 6):
            axes[idx].axis('off')
        
        plt.suptitle('Saliency Maps for Example Molecules', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('attention_figures/summary_grid.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    
    if results:
        print(f"\nSuccessfully processed {len(results)} molecules:")
        for r in results:
            print(f"  - {r['name']}: {r['num_atoms']} atoms")
    else:
        print("\nNo molecules were successfully processed!")
        print("Please check if RDKit is properly installed.")
    
    print("\nOutput files saved to: attention_figures/")
    
    return results


if __name__ == "__main__":
    results = main()