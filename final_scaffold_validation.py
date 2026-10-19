"""
FINAL: Multi-Dataset Scaffold Split Validation for Q1 Paper
Complete pipeline with validated results

Run: python final_scaffold_validation.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import pandas as pd
import numpy as np
import warnings
import random
import json
from datetime import datetime
warnings.filterwarnings('ignore')

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

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
            atomic_num = atom.GetAtomicNum()
            
            # One-hot for common atom types
            if atomic_num == 6:      # Carbon
                feat[0] = 1.0
            elif atomic_num == 7:    # Nitrogen
                feat[1] = 1.0
            elif atomic_num == 8:    # Oxygen
                feat[2] = 1.0
            elif atomic_num == 9:    # Fluorine
                feat[3] = 1.0
            elif atomic_num == 16:   # Sulfur
                feat[4] = 1.0
            elif atomic_num == 17:   # Chlorine
                feat[5] = 1.0
            else:
                feat[6] = 1.0
            
            # Additional atom properties
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
    
    def train(self, train_loader, val_loader, epochs=100, patience=20):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics, _, _ = self.evaluate(val_loader)
            
            if val_metrics['rmse'] < self.best_val_loss:
                self.best_val_loss = val_metrics['rmse']
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    break
            
            if epoch % 25 == 0:
                print(f"    Epoch {epoch:3d} | Loss: {train_loss:.4f} | Val RMSE: {val_metrics['rmse']:.4f}")
        
        self.model.load_state_dict(torch.load('best_model.pt'))
        return self.evaluate(val_loader)


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_dataset(name, smiles, y, device):
    print(f"\n  {'='*50}")
    print(f"  {name}")
    print(f"  {'='*50}")
    
    # Statistics
    print(f"  Molecules: {len(smiles)}")
    print(f"  Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Normalize
    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / y_std
    
    # Scaffold split
    train_idx, val_idx, test_idx = scaffold_split(smiles)
    print(f"  Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    # Create datasets
    train_dataset = MoleculeGraphDataset(smiles[train_idx], y_norm[train_idx])
    val_dataset = MoleculeGraphDataset(smiles[val_idx], y_norm[val_idx])
    test_dataset = MoleculeGraphDataset(smiles[test_idx], y_norm[test_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Train
    print("  Training...")
    model = GCNModel()
    trainer = Trainer(model, device, lr=0.001)
    val_metrics, _, _ = trainer.train(train_loader, val_loader, epochs=100)
    
    # Test
    test_metrics, test_preds, test_targets = trainer.evaluate(test_loader)
    
    # Convert to original scale
    test_preds_orig = test_preds * y_std + y_mean
    test_targets_orig = y[test_idx]
    
    rmse = np.sqrt(mean_squared_error(test_targets_orig, test_preds_orig))
    mae = mean_absolute_error(test_targets_orig, test_preds_orig)
    r2 = r2_score(test_targets_orig, test_preds_orig)
    
    print(f"\n  Results:")
    print(f"    R²:  {r2:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAE:  {mae:.4f}")
    
    return {
        'dataset': name,
        'n_molecules': len(smiles),
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx),
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'target_mean': float(y_mean),
        'target_std': float(y_std)
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("FINAL SCAFFOLD SPLIT VALIDATION - Q1 PAPER RESULTS")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load and evaluate datasets
    print("\n[1/2] Loading datasets...")
    results = []
    
    # ESOL
    smiles, y = load_esol()
    results.append(evaluate_dataset('ESOL', smiles, y, device))
    
    # Lipophilicity
    smiles, y = load_lipophilicity()
    results.append(evaluate_dataset('Lipophilicity', smiles, y, device))
    
    # Summary
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
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()