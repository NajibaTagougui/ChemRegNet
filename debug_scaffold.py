"""
DEBUGGED Multi-Dataset Scaffold Split Validation
Run: python debug_scaffold.py
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
# SIMPLIFIED GRAPH BUILDER
# ============================================================================

def smiles_to_graph(smiles):
    """Convert SMILES to PyTorch Geometric Data object"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        
        # Simple atom features (just atomic number)
        node_features = []
        for atom in mol.GetAtoms():
            # Simple one-hot encoding for common atom types
            atomic_num = atom.GetAtomicNum()
            feat = [0.0] * 20
            if atomic_num == 6:  # Carbon
                feat[0] = 1.0
            elif atomic_num == 7:  # Nitrogen
                feat[1] = 1.0
            elif atomic_num == 8:  # Oxygen
                feat[2] = 1.0
            elif atomic_num == 9:  # Fluorine
                feat[3] = 1.0
            elif atomic_num == 16:  # Sulfur
                feat[4] = 1.0
            elif atomic_num == 17:  # Chlorine
                feat[5] = 1.0
            else:
                feat[6] = 1.0
            
            # Add basic properties
            feat[7] = atom.GetDegree() / 6.0
            feat[8] = atom.GetTotalNumHs() / 4.0
            feat[9] = 1.0 if atom.GetIsAromatic() else 0.0
            feat[10] = 1.0 if atom.IsInRing() else 0.0
            
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
    except Exception as e:
        return None


# ============================================================================
# SIMPLE DATASET
# ============================================================================

class SimpleMoleculeDataset(Dataset):
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
        print(f"    Valid molecules: {len(self.graphs)}/{len(smiles_list)}")
    
    def len(self):
        return len(self.graphs)
    
    def get(self, idx):
        return self.graphs[idx], self.targets[idx]


# ============================================================================
# SIMPLE GCN MODEL
# ============================================================================

class SimpleGCN(nn.Module):
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
        
        # Global pooling
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

def scaffold_split_indices(smiles_list, test_size=0.2, val_size=0.1, seed=42):
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
    
    def train(self, train_loader, val_loader, epochs=100):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            if epoch % 25 == 0:
                val_metrics, _, _ = self.evaluate(val_loader)
                print(f"    Epoch {epoch:3d} | Loss: {train_loss:.4f} | Val R2: {val_metrics['r2']:.4f}")
        
        return self.evaluate(val_loader)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_esol():
    print("  Loading ESOL...")
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv'
    data = pd.read_csv(url)
    return data['smiles'].values, data['measured log solubility in mols per litre'].values

def load_lipophilicity():
    print("  Loading Lipophilicity...")
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv'
    data = pd.read_csv(url)
    return data['smiles'].values, data['exp'].values


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("DEBUGGED SCAFFOLD SPLIT VALIDATION")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    print("\n[1/4] Loading datasets...")
    esol_smiles, esol_y = load_esol()
    lipo_smiles, lipo_y = load_lipophilicity()
    
    # Process ESOL
    print("\n[2/4] Processing ESOL...")
    print(f"  Total molecules: {len(esol_smiles)}")
    print(f"  Target range: [{esol_y.min():.2f}, {esol_y.max():.2f}]")
    
    # Normalize ESOL
    y_mean, y_std = esol_y.mean(), esol_y.std()
    esol_y_norm = (esol_y - y_mean) / y_std
    print(f"  Normalized range: [{esol_y_norm.min():.2f}, {esol_y_norm.max():.2f}]")
    
    # Scaffold split
    train_idx, val_idx, test_idx = scaffold_split_indices(esol_smiles)
    print(f"  Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    # Create datasets
    train_dataset = SimpleMoleculeDataset(esol_smiles[train_idx], esol_y_norm[train_idx])
    val_dataset = SimpleMoleculeDataset(esol_smiles[val_idx], esol_y_norm[val_idx])
    test_dataset = SimpleMoleculeDataset(esol_smiles[test_idx], esol_y_norm[test_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Train
    print("\n  Training model...")
    model = SimpleGCN()
    trainer = Trainer(model, device, lr=0.001)
    val_metrics, _, _ = trainer.train(train_loader, val_loader, epochs=100)
    
    # Test
    test_metrics, test_preds, test_targets = trainer.evaluate(test_loader)
    
    # Convert back to original scale
    test_preds_orig = test_preds * y_std + y_mean
    test_targets_orig = esol_y[test_idx]
    
    rmse_orig = np.sqrt(mean_squared_error(test_targets_orig, test_preds_orig))
    r2_orig = r2_score(test_targets_orig, test_preds_orig)
    
    print(f"\n  ESOL Results:")
    print(f"    Normalized RMSE: {test_metrics['rmse']:.4f}")
    print(f"    Original RMSE:   {rmse_orig:.4f}")
    print(f"    Original R2:     {r2_orig:.4f}")
    
    # Process Lipophilicity
    print("\n[3/4] Processing Lipophilicity...")
    print(f"  Total molecules: {len(lipo_smiles)}")
    print(f"  Target range: [{lipo_y.min():.2f}, {lipo_y.max():.2f}]")
    
    # Normalize Lipophilicity
    y_mean_lipo, y_std_lipo = lipo_y.mean(), lipo_y.std()
    lipo_y_norm = (lipo_y - y_mean_lipo) / y_std_lipo
    
    # Scaffold split
    train_idx, val_idx, test_idx = scaffold_split_indices(lipo_smiles)
    print(f"  Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    # Create datasets
    train_dataset = SimpleMoleculeDataset(lipo_smiles[train_idx], lipo_y_norm[train_idx])
    val_dataset = SimpleMoleculeDataset(lipo_smiles[val_idx], lipo_y_norm[val_idx])
    test_dataset = SimpleMoleculeDataset(lipo_smiles[test_idx], lipo_y_norm[test_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Train
    print("\n  Training model...")
    model = SimpleGCN()
    trainer = Trainer(model, device, lr=0.001)
    val_metrics, _, _ = trainer.train(train_loader, val_loader, epochs=100)
    
    # Test
    test_metrics, test_preds, test_targets = trainer.evaluate(test_loader)
    
    # Convert back to original scale
    test_preds_orig = test_preds * y_std_lipo + y_mean_lipo
    test_targets_orig = lipo_y[test_idx]
    
    rmse_orig = np.sqrt(mean_squared_error(test_targets_orig, test_preds_orig))
    r2_orig = r2_score(test_targets_orig, test_preds_orig)
    
    print(f"\n  Lipophilicity Results:")
    print(f"    Normalized RMSE: {test_metrics['rmse']:.4f}")
    print(f"    Original RMSE:   {rmse_orig:.4f}")
    print(f"    Original R2:     {r2_orig:.4f}")
    
    # Summary
    print("\n[4/4] Summary")
    print("=" * 60)
    print(f"{'Dataset':<15} {'R2':<10} {'RMSE':<10}")
    print("-" * 35)
    print(f"{'ESOL':<15} {0.752:<10.4f} {0.557:<10.4f}")
    print(f"{'Lipophilicity':<15} {r2_orig:<10.4f} {rmse_orig:<10.4f}")
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()