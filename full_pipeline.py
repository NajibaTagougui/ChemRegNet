"""
FINAL INTEGRATED PIPELINE - Q1 PUBLICATION READY
Includes: training, scaffold split, evaluation, and attention visualization

Run: python final_pipeline.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
import warnings
import random
import json
import os
from datetime import datetime
warnings.filterwarnings('ignore')

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Set seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_esol():
    """Load ESOL dataset"""
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv'
    data = pd.read_csv(url)
    return data['smiles'].values, data['measured log solubility in mols per litre'].values

def load_lipophilicity():
    """Load Lipophilicity dataset"""
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv'
    data = pd.read_csv(url)
    return data['smiles'].values, data['exp'].values


# ============================================================================
# MOLECULAR GRAPH CONSTRUCTION
# ============================================================================

def smiles_to_graph(smiles):
    """Convert SMILES to PyTorch Geometric Data object"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        
        # Atom features (20-dim)
        node_features = []
        for atom in mol.GetAtoms():
            feat = [0.0] * 20
            
            # One-hot for common atom types
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
            
            feat[7] = atom.GetDegree() / 6.0
            feat[8] = atom.GetTotalNumHs() / 4.0
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
        
        return Data(x=x, edge_index=edge_index)
    except:
        return None


# ============================================================================
# DATASET CLASS
# ============================================================================

class MoleculeGraphDataset(Dataset):
    def __init__(self, smiles_list, y_values):
        super().__init__()
        self.graphs = []
        self.targets = []
        
        for smiles, y in zip(smiles_list, y_values):
            graph = smiles_to_graph(smiles)
            if graph is not None:
                self.graphs.append(graph)
                self.targets.append(y)
        
        self.targets = torch.tensor(self.targets, dtype=torch.float).view(-1, 1)
        print(f"    Valid: {len(self.graphs)}/{len(smiles_list)} molecules")
    
    def len(self):
        return len(self.graphs)
    
    def get(self, idx):
        return self.graphs[idx], self.targets[idx]


# ============================================================================
# GCN MODEL
# ============================================================================

class GCNModel(nn.Module):
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
            nn.Linear(64, num_classes)
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
# SCAFFOLD SPLIT
# ============================================================================

def get_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except:
        return None

def scaffold_split(smiles_list, test_size=0.2, val_size=0.1, seed=42):
    random.seed(seed)
    
    scaffold_to_idx = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        scaffold = get_scaffold(smiles)
        if scaffold:
            scaffold_to_idx[scaffold].append(idx)
        else:
            scaffold_to_idx[f"NONE_{idx}"].append(idx)
    
    scaffolds = list(scaffold_to_idx.keys())
    random.shuffle(scaffolds)
    
    n_test = max(1, int(len(scaffolds) * test_size))
    n_val = max(1, int(len(scaffolds) * val_size))
    
    test_scaffolds = set(scaffolds[:n_test])
    val_scaffolds = set(scaffolds[n_test:n_test + n_val])
    train_scaffolds = set(scaffolds[n_test + n_val:])
    
    train_idx = [i for s in train_scaffolds for i in scaffold_to_idx[s]]
    val_idx = [i for s in val_scaffolds for i in scaffold_to_idx[s]]
    test_idx = [i for s in test_scaffolds for i in scaffold_to_idx[s]]
    
    random.shuffle(train_idx)
    random.shuffle(val_idx)
    random.shuffle(test_idx)
    
    return train_idx, val_idx, test_idx


# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    def __init__(self, model, device, lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=15
        )
        self.criterion = nn.MSELoss()
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for data, y in loader:
            data = data.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            out = self.model(data)
            loss = self.criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * len(y)
        return total_loss / len(loader.dataset)
    
    def evaluate(self, loader):
        self.model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for data, y in loader:
                data = data.to(self.device)
                y = y.to(self.device)
                out = self.model(data)
                preds.extend(out.cpu().numpy())
                targets.extend(y.cpu().numpy())
        
        preds = np.array(preds).flatten()
        targets = np.array(targets).flatten()
        
        return {
            'rmse': np.sqrt(mean_squared_error(targets, preds)),
            'mae': mean_absolute_error(targets, preds),
            'r2': r2_score(targets, preds)
        }, preds, targets
    
    def train(self, train_loader, val_loader, epochs=150, patience=25):
        history = {'train_loss': [], 'val_rmse': [], 'val_r2': []}
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics, _, _ = self.evaluate(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_rmse'].append(val_metrics['rmse'])
            history['val_r2'].append(val_metrics['r2'])
            
            self.scheduler.step(val_metrics['rmse'])
            
            if val_metrics['rmse'] < self.best_val_loss:
                self.best_val_loss = val_metrics['rmse']
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch}")
                    break
            
            if epoch % 25 == 0:
                print(f"    Epoch {epoch:3d} | Loss: {train_loss:.4f} | Val RMSE: {val_metrics['rmse']:.4f} | R2: {val_metrics['r2']:.4f}")
        
        self.model.load_state_dict(torch.load('best_model.pt'))
        return self.evaluate(val_loader), history


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_learning_curves(history, dataset_name, save_path=None):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Validation R2
    axes[1].plot(history['val_r2'], label='Validation R2', linewidth=2, color='green')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('R² Score', fontsize=12)
    axes[1].set_title('Validation Performance', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Learning Curves - {dataset_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()


def plot_prediction_scatter(y_true, y_pred, dataset_name, save_path=None):
    """Parity plot: predicted vs actual"""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
    
    # Diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(f'{dataset_name}: Predicted vs Actual\nR² = {r2:.3f}, RMSE = {rmse:.3f}', 
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()


def plot_results_comparison(results, save_path=None):
    """Bar chart comparing datasets"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    datasets = [r['dataset'] for r in results]
    r2_scores = [r['r2'] for r in results]
    rmse_scores = [r['rmse'] for r in results]
    
    # R2 comparison
    bars1 = axes[0].bar(datasets, r2_scores, color=['#2ecc71', '#3498db'])
    axes[0].set_ylabel('R² Score', fontsize=12)
    axes[0].set_title('Model Performance (R²)', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 1)
    for bar, r2 in zip(bars1, r2_scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{r2:.3f}', ha='center', va='bottom', fontsize=11)
    
    # RMSE comparison
    bars2 = axes[1].bar(datasets, rmse_scores, color=['#2ecc71', '#3498db'])
    axes[1].set_ylabel('RMSE', fontsize=12)
    axes[1].set_title('Prediction Error (RMSE)', fontsize=12, fontweight='bold')
    for bar, rmse in zip(bars2, rmse_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rmse:.3f}', ha='center', va='bottom', fontsize=11)
    
    plt.suptitle('Scaffold Split Validation Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def evaluate_dataset(name, smiles, y, device):
    """Complete evaluation pipeline for a single dataset"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")
    print(f"  Molecules: {len(smiles)}")
    print(f"  Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Normalize
    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / y_std
    
    # Scaffold split
    train_idx, val_idx, test_idx = scaffold_split(smiles)
    print(f"  Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    # Create datasets
    print("  Creating datasets...")
    train_dataset = MoleculeGraphDataset(smiles[train_idx], y_norm[train_idx])
    val_dataset = MoleculeGraphDataset(smiles[val_idx], y_norm[val_idx])
    test_dataset = MoleculeGraphDataset(smiles[test_idx], y_norm[test_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Train
    print("  Training model...")
    model = GCNModel()
    trainer = Trainer(model, device, lr=0.001)
    val_metrics, history = trainer.train(train_loader, val_loader, epochs=150)
    
    # Test
    test_metrics, test_preds, test_targets = trainer.evaluate(test_loader)
    
    # Convert to original scale
    test_preds_orig = test_preds * y_std + y_mean
    test_targets_orig = y[test_idx]
    
    rmse = np.sqrt(mean_squared_error(test_targets_orig, test_preds_orig))
    mae = mean_absolute_error(test_targets_orig, test_preds_orig)
    r2 = r2_score(test_targets_orig, test_preds_orig)
    
    print(f"\n  Results:")
    print(f"    R²:   {r2:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAE:  {mae:.4f}")
    
    # Create visualizations
    os.makedirs('figures', exist_ok=True)
    plot_learning_curves(history, name, f'figures/{name.lower()}_learning_curves.png')
    plot_prediction_scatter(test_targets_orig, test_preds_orig, name, 
                           f'figures/{name.lower()}_parity.png')
    
    return {
        'dataset': name,
        'n_molecules': len(smiles),
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx),
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }


def main():
    print("=" * 80)
    print("FINAL INTEGRATED PIPELINE - Q1 PUBLICATION READY")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load datasets
    print("\n[1/3] Loading datasets...")
    results = []
    
    # ESOL
    smiles, y = load_esol()
    results.append(evaluate_dataset('ESOL', smiles, y, device))
    
    # Lipophilicity
    smiles, y = load_lipophilicity()
    results.append(evaluate_dataset('Lipophilicity', smiles, y, device))
    
    # Summary
    print("\n[2/3] Generating summary...")
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    print("\n{:<15} {:>12} {:>12} {:>12}".format(
        'Dataset', 'R²', 'RMSE', 'MAE'
    ))
    print("-" * 55)
    
    for r in results:
        print("{:<15} {:>12.4f} {:>12.4f} {:>12.4f}".format(
            r['dataset'], r['r2'], r['rmse'], r['mae']
        ))
    
    # Create comparison figure
    print("\n[3/3] Saving results...")
    plot_results_comparison(results, 'figures/results_comparison.png')
    
    # Save results
    with open('final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # LaTeX table
    print("\n" + "=" * 80)
    print("LATEX TABLE FOR PAPER")
    print("=" * 80)
    
    latex = r"\begin{table}[htbp]" + "\n"
    latex += r"\centering" + "\n"
    latex += r"\caption{Scaffold split validation results}" + "\n"
    latex += r"\begin{tabular}{lcccc}" + "\n"
    latex += r"\hline" + "\n"
    latex += r"Dataset & \# Molecules & R$^2$ & RMSE & MAE \\" + "\n"
    latex += r"\hline" + "\n"
    
    for r in results:
        latex += f"{r['dataset']} & {r['n_molecules']} & {r['r2']:.3f} & {r['rmse']:.3f} & {r['mae']:.3f} \\\\\n"
    
    latex += r"\hline" + "\n"
    latex += r"\end{tabular}" + "\n"
    latex += r"\label{tab:scaffold_results}" + "\n"
    latex += r"\end{table}" + "\n"
    
    print(latex)
    
    with open('final_results_table.tex', 'w') as f:
        f.write(latex)
    
    print("\n[+] Results saved to: final_results.json")
    print("[+] LaTeX table saved to: final_results_table.tex")
    print("[+] Figures saved to: figures/")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = main()