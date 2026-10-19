"""
PUBLICATION-QUALITY ATTENTION VISUALIZATION
High-resolution figures + Custom molecule selection from dataset

Run: python publication_attention.py
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
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Publication-quality settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

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
# MOLECULAR GRAPH CONSTRUCTION
# ============================================================================

def smiles_to_graph_with_coords(smiles):
    """Convert SMILES to graph with 2D coordinates"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, None
        
        try:
            Chem.SanitizeMol(mol)
        except:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return None, None, None
        
        try:
            Chem.rdDepictor.Compute2DCoords(mol)
        except:
            from rdkit.Chem import AllChem
            AllChem.Compute2DCoords(mol)
        
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
        return None, None, None


# ============================================================================
# ATTENTION COMPUTATION
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
# PUBLICATION-QUALITY VISUALIZATIONS
# ============================================================================

def visualize_publication_attention(mol, positions, attention_scores, title, 
                                     save_path=None, cmap_name='RdYlGn_r', 
                                     figsize=(12, 10), dpi=300):
    """
    High-resolution publication-quality attention visualization
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    scores = np.array(attention_scores)
    if scores.max() > 0:
        scores_norm = scores / scores.max()
    else:
        scores_norm = scores
    
    cmap = cm.get_cmap(cmap_name)
    
    # Draw bonds with varying thickness
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i < len(positions) and j < len(positions):
            x1, y1 = positions[i]
            x2, y2 = positions[j]
            
            bt = bond.GetBondType()
            if bt == Chem.rdchem.BondType.SINGLE:
                linewidth = 2.5
                linestyle = '-'
            elif bt == Chem.rdchem.BondType.DOUBLE:
                linewidth = 3.0
                linestyle = '-'
            elif bt == Chem.rdchem.BondType.TRIPLE:
                linewidth = 3.5
                linestyle = '-'
            else:
                linewidth = 2.0
                linestyle = '-'
            
            ax.plot([x1, x2], [y1, y2], color='#333333', linewidth=linewidth, 
                   linestyle=linestyle, zorder=1, alpha=0.8)
    
    # Draw atoms with attention-based coloring
    for i, (x, y) in enumerate(positions):
        if i < len(scores_norm):
            color = cmap(scores_norm[i])
            size = 1200 * (0.3 + scores_norm[i] * 0.7)
            edge_color = 'black' if scores_norm[i] > 0.5 else '#666666'
            edge_width = 2.0 if scores_norm[i] > 0.5 else 1.5
        else:
            color = cmap(0)
            size = 600
            edge_color = '#666666'
            edge_width = 1.5
        
        ax.scatter(x, y, s=size, c=[color], edgecolors=edge_color, 
                  linewidth=edge_width, zorder=2)
        
        if i < mol.GetNumAtoms():
            # Get atom label with subscript formatting for publication
            symbol = mol.GetAtomWithIdx(i).GetSymbol()
            # Convert to proper chemical notation
            if len(symbol) > 1:
                symbol = symbol[0] + '\\' + symbol[1]  # For LaTeX subscript
            ax.text(x, y, symbol, ha='center', va='center', 
                   fontsize=14, fontweight='bold', color='black', zorder=3)
    
    # Set axes with padding
    if positions:
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        padding = 1.5
        ax.set_xlim(min(x_coords) - padding, max(x_coords) + padding)
        ax.set_ylim(min(y_coords) - padding, max(y_coords) + padding)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add colorbar with publication styling
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=25, pad=0.02)
    cbar.set_label('Attention Score', fontsize=12, labelpad=10)
    cbar.ax.tick_params(labelsize=10)
    
    # Add title with molecule name
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add subtle bounding box
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"    Saved: {save_path}")
    
    plt.close()
    return fig


def visualize_attention_comparison_publication(mol, positions, attention_methods, 
                                                save_path=None, dpi=300):
    """
    Publication-quality comparison of attention methods
    """
    n_methods = len(attention_methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5), dpi=dpi)
    
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
        
        # Draw bonds
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if i < len(positions) and j < len(positions):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                ax.plot([x1, x2], [y1, y2], '#333333', linewidth=2, zorder=1, alpha=0.7)
        
        # Draw atoms
        for i, (x, y) in enumerate(positions):
            if i < len(scores_norm):
                color = cmap(scores_norm[i])
                size = 800 * (0.3 + scores_norm[i] * 0.7)
                ax.scatter(x, y, s=size, c=[color], edgecolors='black', 
                          linewidth=1.5, zorder=2)
                if i < mol.GetNumAtoms():
                    ax.text(x, y, mol.GetAtomWithIdx(i).GetSymbol(), 
                           ha='center', va='center', fontsize=11, 
                           fontweight='bold', zorder=3)
        
        if positions:
            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            padding = 1.0
            ax.set_xlim(min(x_coords) - padding, max(x_coords) + padding)
            ax.set_ylim(min(y_coords) - padding, max(y_coords) + padding)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(method_name, fontsize=12, fontweight='bold')
    
    plt.suptitle(f'{mol.GetProp("_Name") if mol.HasProp("_Name") else "Molecule"}: Attention Method Comparison', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"    Saved: {save_path}")
    
    plt.close()


def create_publication_figure_grid(results, save_path=None, dpi=300):
    """
    Create a publication-ready figure grid of multiple molecules
    """
    n_molecules = len(results)
    cols = min(3, n_molecules)
    rows = (n_molecules + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), dpi=dpi)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    axes = axes.flatten()
    
    for idx, result in enumerate(results):
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
                ax.plot([x1, x2], [y1, y2], '#333333', linewidth=2, zorder=1, alpha=0.7)
        
        # Draw atoms
        for i, (x, y) in enumerate(positions):
            if i < len(saliency_norm):
                color = cmap(saliency_norm[i])
                size = 500 * (0.3 + saliency_norm[i] * 0.7)
                ax.scatter(x, y, s=size, c=[color], edgecolors='black', 
                          linewidth=1.2, zorder=2)
                ax.text(x, y, mol.GetAtomWithIdx(i).GetSymbol(), 
                       ha='center', va='center', fontsize=10, fontweight='bold', zorder=3)
        
        if positions:
            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            padding = 1.0
            ax.set_xlim(min(x_coords) - padding, max(x_coords) + padding)
            ax.set_ylim(min(y_coords) - padding, max(y_coords) + padding)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add molecule name and SMILES
        name = result['name']
        ax.set_title(f"{name}\n{smiles[:40]}{'...' if len(smiles) > 40 else ''}", 
                    fontsize=9, fontweight='bold')
    
    # Hide empty subplots
    for idx in range(n_molecules, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Saliency Maps for Selected Molecules', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    plt.close()


# ============================================================================
# DATASET LOADING FOR CUSTOM MOLECULES
# ============================================================================

def load_dataset_data():
    """Load ESOL and Lipophilicity datasets"""
    datasets = {}
    
    # ESOL
    try:
        url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv'
        esol_data = pd.read_csv(url)
        datasets['ESOL'] = {
            'smiles': esol_data['smiles'].values,
            'targets': esol_data['measured log solubility in mols per litre'].values,
            'target_name': 'logS'
        }
        print(f"  Loaded ESOL: {len(datasets['ESOL']['smiles'])} molecules")
    except Exception as e:
        print(f"  Could not load ESOL: {e}")
    
    # Lipophilicity
    try:
        url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv'
        lipo_data = pd.read_csv(url)
        datasets['Lipophilicity'] = {
            'smiles': lipo_data['smiles'].values,
            'targets': lipo_data['exp'].values,
            'target_name': 'logD'
        }
        print(f"  Loaded Lipophilicity: {len(datasets['Lipophilicity']['smiles'])} molecules")
    except Exception as e:
        print(f"  Could not load Lipophilicity: {e}")
    
    return datasets


def select_molecules_by_property(datasets, dataset_name, property_range=None, n_molecules=5):
    """
    Select molecules based on property range
    """
    if dataset_name not in datasets:
        return []
    
    data = datasets[dataset_name]
    smiles_list = data['smiles']
    targets = data['targets']
    
    selected = []
    for smiles, target in zip(smiles_list, targets):
        if property_range:
            low, high = property_range
            if low <= target <= high:
                selected.append((smiles, target))
        else:
            selected.append((smiles, target))
        
        if len(selected) >= n_molecules:
            break
    
    return selected


def get_high_interest_molecules():
    """
    Predefined list of high-interest drug molecules
    """
    molecules = {
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
        "Naproxen": "COC1=CC2=C(C=C1)C(C(=O)O)C=C2",
        "Dopamine": "NCCC1=CC(=C(C=C1)O)O",
        "Serotonin": "C1=CC2=C(C=C1O)NC=C2CCN",
        "Penicillin": "CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C",
        "Morphine": "CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
        "Cholesterol": "CC(C)CCCC(C)C1CCC2C3CCC4CC(O)CCC4(C)C3CCC12C",
    }
    return molecules


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 80)
    print("PUBLICATION-QUALITY ATTENTION VISUALIZATION")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    os.makedirs('publication_figures', exist_ok=True)
    os.makedirs('publication_figures/high_res', exist_ok=True)
    os.makedirs('publication_figures/custom_molecules', exist_ok=True)
    
    # Load model
    print("\n[1/5] Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNModelWithGradients()
    
    model_path = 'best_model.pt'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"  Loaded model from {model_path}")
    else:
        print(f"  WARNING: {model_path} not found! Using untrained model.")
    
    model = model.to(device)
    model.eval()
    
    # Load datasets for custom selection
    print("\n[2/5] Loading datasets...")
    datasets = load_dataset_data()
    
    # Define molecules to visualize
    print("\n[3/5] Defining molecules for visualization...")
    
    # Option A: High-interest drug molecules
    high_interest = get_high_interest_molecules()
    print(f"  High-interest molecules: {len(high_interest)}")
    
    # Option B: Select from dataset by property (example: highly soluble molecules)
    if 'ESOL' in datasets:
        # Select highly soluble molecules (logS > 0)
        highly_soluble = select_molecules_by_property(datasets, 'ESOL', property_range=(0, 5), n_molecules=5)
        print(f"  Highly soluble molecules from ESOL: {len(highly_soluble)}")
        
        # Select poorly soluble molecules (logS < -4)
        poorly_soluble = select_molecules_by_property(datasets, 'ESOL', property_range=(-10, -4), n_molecules=5)
        print(f"  Poorly soluble molecules from ESOL: {len(poorly_soluble)}")
    
    # Process molecules
    print("\n[4/5] Computing attention maps and generating figures...")
    all_results = []
    
    # Process high-interest molecules first
    for name, smiles in high_interest.items():
        print(f"\n  Processing: {name}")
        
        data, positions, mol = smiles_to_graph_with_coords(smiles)
        if data is None or positions is None or mol is None:
            print(f"    Failed to process {name}")
            continue
        
        # Set molecule name for titles
        mol.SetProp("_Name", name)
        
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        
        # Compute attention
        saliency = compute_saliency_attention(model, data, device)
        activation = compute_activation_attention(model, data, device)
        
        # Generate publication-quality figures
        save_name = name.lower().replace(' ', '_')
        
        # High-resolution saliency
        visualize_publication_attention(
            mol, positions, saliency,
            f'{name}: Saliency Map',
            f'publication_figures/high_res/{save_name}_saliency.png',
            figsize=(14, 12), dpi=600
        )
        
        # High-resolution activation
        visualize_publication_attention(
            mol, positions, activation,
            f'{name}: Activation Attention',
            f'publication_figures/high_res/{save_name}_activation.png',
            cmap_name='plasma', figsize=(14, 12), dpi=600
        )
        
        # Comparison
        attention_methods = {'Saliency': saliency, 'Activation': activation}
        visualize_attention_comparison_publication(
            mol, positions, attention_methods,
            f'publication_figures/high_res/{save_name}_comparison.png', dpi=300
        )
        
        all_results.append({
            'name': name,
            'smiles': smiles,
            'num_atoms': mol.GetNumAtoms(),
            'saliency': saliency.tolist(),
            'activation': activation.tolist()
        })
        
        print(f"    SUCCESS! Generated 3 high-res figures for {name}")
    
    # Process dataset-selected molecules
    if 'ESOL' in datasets:
        for prop_type, molecules, label in [
            ('highly_soluble', highly_soluble, 'Highly Soluble'),
            ('poorly_soluble', poorly_soluble, 'Poorly Soluble')
        ]:
            for idx, (smiles, target) in enumerate(molecules[:3]):  # Limit to 3 per category
                name = f"{label}_{idx+1}"
                print(f"\n  Processing: {name} (target={target:.2f})")
                
                data, positions, mol = smiles_to_graph_with_coords(smiles)
                if data is None or positions is None or mol is None:
                    print(f"    Failed to process")
                    continue
                
                mol.SetProp("_Name", name)
                data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
                
                saliency = compute_saliency_attention(model, data, device)
                
                save_name = f"{prop_type}_{idx+1}"
                visualize_publication_attention(
                    mol, positions, saliency,
                    f'{label} Molecule (target={target:.2f})',
                    f'publication_figures/custom_molecules/{save_name}_saliency.png',
                    figsize=(12, 10), dpi=300
                )
                
                all_results.append({
                    'name': name,
                    'smiles': smiles,
                    'target': float(target),
                    'num_atoms': mol.GetNumAtoms(),
                    'saliency': saliency.tolist()
                })
    
    # Create publication grid
    print("\n[5/5] Creating publication grid...")
    if all_results:
        create_publication_figure_grid(
            all_results[:9],  # Limit to 9 for grid
            'publication_figures/publication_grid.png',
            dpi=300
        )
    
    # Save all results
    with open('publication_figures/attention_results_complete.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("PUBLICATION VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nSuccessfully processed {len(all_results)} molecules")
    print("\nOutput directories:")
    print("  - publication_figures/high_res/       (600 DPI figures for publication)")
    print("  - publication_figures/custom_molecules/ (Dataset-specific molecules)")
    print("  - publication_figures/publication_grid.png (Summary grid)")
    print("  - publication_figures/attention_results_complete.json (Numerical data)")
    
    return all_results


if __name__ == "__main__":
    results = main()