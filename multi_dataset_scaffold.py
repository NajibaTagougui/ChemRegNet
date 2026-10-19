"""
Multi-Dataset Scaffold Split Validation (OPTIMIZED)
With dataset-specific hyperparameters

Run: python multi_dataset_optimized.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
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

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Set random seeds
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
    print("  Loading ESOL dataset...")
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv'
    data = pd.read_csv(url)
    smiles = data['smiles'].values
    y = data['measured log solubility in mols per litre'].values
    print(f"    Loaded {len(smiles)} molecules, range: [{y.min():.2f}, {y.max():.2f}]")
    return smiles, y, 'ESOL'

def load_lipophilicity():
    """Load Lipophilicity dataset"""
    print("  Loading Lipophilicity dataset...")
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv'
    data = pd.read_csv(url)
    smiles = data['smiles'].values
    y = data['exp'].values
    print(f"    Loaded {len(smiles)} molecules, range: [{y.min():.2f}, {y.max():.2f}]")
    return smiles, y, 'Lipophilicity'


# ============================================================================
# SCAFFOLD SPLIT
# ============================================================================

def generate_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return scaffold
    except:
        return None

def scaffold_split(smiles_list, y_values, test_size=0.2, val_size=0.1, seed=42):
    random.seed(seed)
    
    scaffold_to_indices = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles)
        if scaffold is not None:
            scaffold_to_indices[scaffold].append(idx)
        else:
            scaffold_to_indices[f"NO_SCAFFOLD_{idx}"].append(idx)
    
    scaffolds = list(scaffold_to_indices.keys())
    random.shuffle(scaffolds)
    
    n_test = max(1, int(len(scaffolds) * test_size))
    n_val = max(1, int(len(scaffolds) * val_size))
    
    test_scaffolds = set(scaffolds[:n_test])
    val_scaffolds = set(scaffolds[n_test:n_test + n_val])
    train_scaffolds = set(scaffolds[n_test + n_val:])
    
    train_idx = [i for s in train_scaffolds for i in scaffold_to_indices[s]]
    val_idx = [i for s in val_scaffolds for i in scaffold_to_indices[s]]
    test_idx = [i for s in test_scaffolds for i in scaffold_to_indices[s]]
    
    random.shuffle(train_idx)
    random.shuffle(val_idx)
    random.shuffle(test_idx)
    
    return train_idx, val_idx, test_idx


# ============================================================================
# MOLECULAR GRAPH BUILDER
# ============================================================================

class MolecularGraphBuilder:
    @staticmethod
    def get_atom_features(atom, mol):
        features = []
        features.append(min(atom.GetAtomicNum() / 100.0, 1.0))
        features.append(min(atom.GetDegree(), 5) / 5.0)
        features.append((atom.GetFormalCharge() + 1) / 3.0)
        features.append(min(atom.GetTotalNumHs(), 4) / 4.0)
        hyb = atom.GetHybridization()
        features.append(1.0 if hyb == Chem.rdchem.HybridizationType.SP else 0.0)
        features.append(1.0 if hyb == Chem.rdchem.HybridizationType.SP2 else 0.0)
        features.append(1.0 if hyb == Chem.rdchem.HybridizationType.SP3 else 0.0)
        features.append(1.0 if atom.GetIsAromatic() else 0.0)
        is_in_ring = atom.IsInRing()
        features.append(1.0 if is_in_ring else 0.0)
        if is_in_ring:
            ring_info = mol.GetRingInfo()
            ring_sizes = ring_info.AtomRingSizes(atom.GetIdx())
            features.append(min(ring_sizes[0], 8) / 8.0 if ring_sizes else 0.0)
        else:
            features.append(0.0)
        chiral = atom.GetChiralTag()
        features.append(1.0 if chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW else 0.0)
        features.append(1.0 if chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW else 0.0)
        features.append(min(atom.GetMass(), 200) / 200.0)
        while len(features) < 20:
            features.append(0.0)
        return np.array(features[:20], dtype=np.float32)
    
    @staticmethod
    def get_bond_features(bond):
        features = []
        bond_type = bond.GetBondType()
        features.append(1.0 if bond_type == Chem.rdchem.BondType.SINGLE else 0.0)
        features.append(1.0 if bond_type == Chem.rdchem.BondType.DOUBLE else 0.0)
        features.append(1.0 if bond_type == Chem.rdchem.BondType.TRIPLE else 0.0)
        features.append(1.0 if bond_type == Chem.rdchem.BondType.AROMATIC else 0.0)
        features.append(1.0 if bond.GetIsConjugated() else 0.0)
        features.append(1.0 if bond.IsInRing() else 0.0)
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def mol_to_graph(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            Chem.SanitizeMol(mol)
            
            node_features = []
            for atom in mol.GetAtoms():
                node_features.append(MolecularGraphBuilder.get_atom_features(atom, mol))
            
            edge_indices = []
            edge_features = []
            
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_indices.append([i, j])
                edge_indices.append([j, i])
                bond_feat = MolecularGraphBuilder.get_bond_features(bond)
                edge_features.append(bond_feat)
                edge_features.append(bond_feat)
            
            if len(edge_indices) == 0:
                edge_indices = [[0, 0]]
                edge_features = [np.zeros(6, dtype=np.float32)]
            
            x = torch.tensor(np.array(node_features), dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float)
            
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        except:
            return None


# ============================================================================
# DESCRIPTOR EXTRACTOR
# ============================================================================

class MolecularDescriptorExtractor:
    @staticmethod
    def compute_descriptors(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(50, dtype=np.float32)
            Chem.SanitizeMol(mol)
            
            descriptors = []
            descriptors.append(Descriptors.MolWt(mol))
            descriptors.append(Descriptors.MolLogP(mol))
            descriptors.append(Descriptors.NumHAcceptors(mol))
            descriptors.append(Descriptors.NumHDonors(mol))
            descriptors.append(Descriptors.TPSA(mol))
            descriptors.append(Descriptors.NumRotatableBonds(mol))
            descriptors.append(Descriptors.NumAromaticRings(mol))
            descriptors.append(Descriptors.NumAliphaticRings(mol))
            descriptors.append(Descriptors.NumSaturatedRings(mol))
            descriptors.append(Descriptors.NumHeteroatoms(mol))
            descriptors.append(Descriptors.FractionCSP3(mol))
            descriptors.append(Descriptors.BalabanJ(mol))
            descriptors.append(Descriptors.BertzCT(mol))
            descriptors.append(Descriptors.HallKierAlpha(mol))
            descriptors.append(Descriptors.Kappa1(mol))
            descriptors.append(Descriptors.Kappa2(mol))
            descriptors.append(Descriptors.Kappa3(mol))
            descriptors.append(Descriptors.HeavyAtomCount(mol))
            descriptors.append(Descriptors.HeavyAtomMolWt(mol))
            descriptors.append(Descriptors.NHOHCount(mol))
            descriptors.append(Descriptors.NOCount(mol))
            descriptors.append(rdMolDescriptors.CalcTPSA(mol))
            descriptors.append(Descriptors.MolMR(mol))
            descriptors.append(Descriptors.ExactMolWt(mol))
            
            while len(descriptors) < 50:
                descriptors.append(0.0)
            
            return np.array(descriptors[:50], dtype=np.float32)
        except:
            return np.zeros(50, dtype=np.float32)


# ============================================================================
# DATASET
# ============================================================================

class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, y_values):
        super().__init__()
        self.valid_graphs = []
        self.valid_descriptors = []
        self.valid_y = []
        
        for smiles, y in zip(smiles_list, y_values):
            graph = MolecularGraphBuilder.mol_to_graph(smiles)
            if graph is None:
                continue
            desc = MolecularDescriptorExtractor.compute_descriptors(smiles)
            self.valid_graphs.append(graph)
            self.valid_descriptors.append(desc)
            self.valid_y.append(y)
        
        self.valid_descriptors = np.array(self.valid_descriptors)
        self.valid_y = torch.tensor(self.valid_y, dtype=torch.float).view(-1, 1)
        
        # Normalize descriptors
        mean = self.valid_descriptors.mean(axis=0)
        std = self.valid_descriptors.std(axis=0)
        self.valid_descriptors = (self.valid_descriptors - mean) / (std + 1e-8)
    
    def len(self):
        return len(self.valid_graphs)
    
    def get(self, idx):
        return self.valid_graphs[idx], torch.tensor(self.valid_descriptors[idx], dtype=torch.float), self.valid_y[idx]


# ============================================================================
# MODEL WITH DROPOUT ADJUSTMENT
# ============================================================================

class HybridGCNModel(nn.Module):
    def __init__(self, node_features_dim=20, hidden_dim=128, descriptor_dim=50, 
                 num_classes=1, dropout=0.3):  # Increased dropout for Lipophilicity
        super().__init__()
        
        self.conv1 = GCNConv(node_features_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        
        self.descriptor_branch = nn.Sequential(
            nn.Linear(descriptor_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, graph_data, descriptors):
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        
        x = F.relu(self.conv1(x, edge_index))
        x = self.batch_norm1(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x, edge_index))
        x = self.batch_norm2(x)
        x = self.dropout(x)
        
        gnn_mean = global_mean_pool(x, batch)
        gnn_max = global_max_pool(x, batch)
        gnn_features = torch.cat([gnn_mean, gnn_max], dim=1)
        
        desc_features = self.descriptor_branch(descriptors)
        
        combined = torch.cat([gnn_features, desc_features], dim=1)
        combined = self.fusion(combined)
        
        return self.predictor(combined)


# ============================================================================
# TRAINER WITH ADAPTIVE LEARNING RATE
# ============================================================================

class ModelTrainer:
    def __init__(self, model, device, learning_rate=0.0005, weight_decay=1e-5):  # Lower LR
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=15)
        self.criterion = nn.MSELoss()
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for graph_data, descriptors, y in loader:
            graph_data = graph_data.to(self.device)
            descriptors = descriptors.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(graph_data, descriptors)
            loss = self.criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(y)
        return total_loss / len(loader.dataset)
    
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for graph_data, descriptors, y in loader:
                graph_data = graph_data.to(self.device)
                descriptors = descriptors.to(self.device)
                y = y.to(self.device)
                output = self.model(graph_data, descriptors)
                loss = self.criterion(output, y)
                total_loss += loss.item() * len(y)
                all_preds.extend(output.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        preds = np.array(all_preds).flatten()
        targets = np.array(all_targets).flatten()
        
        return {
            'loss': total_loss / len(loader.dataset),
            'rmse': np.sqrt(mean_squared_error(targets, preds)),
            'mae': mean_absolute_error(targets, preds),
            'r2': r2_score(targets, preds)
        }, preds, targets
    
    def train(self, train_loader, val_loader, epochs=150, verbose=True):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics, _, _ = self.evaluate(val_loader)
            self.scheduler.step(val_metrics['loss'])
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                self.patience_counter += 1
                if self.patience_counter >= 30:
                    break
            
            if verbose and epoch % 25 == 0:
                print(f"    Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val R2: {val_metrics['r2']:.4f}")
        
        self.model.load_state_dict(torch.load('best_model.pt'))
        return self.evaluate(val_loader)


# ============================================================================
# DATASET-SPECIFIC CONFIGURATIONS
# ============================================================================

def get_dataset_config(dataset_name):
    """Return dataset-specific hyperparameters"""
    configs = {
        'ESOL': {
            'learning_rate': 0.0005,
            'dropout': 0.2,
            'hidden_dim': 128,
            'epochs': 120,
            'patience': 20
        },
        'Lipophilicity': {
            'learning_rate': 0.0003,  # Lower LR for stability
            'dropout': 0.3,           # Higher dropout to prevent overfitting
            'hidden_dim': 256,        # Larger capacity
            'epochs': 150,
            'patience': 25
        }
    }
    return configs.get(dataset_name, configs['ESOL'])


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def evaluate_on_dataset(dataset_name, smiles, y, device):
    """Run scaffold split with dataset-specific config"""
    
    print(f"\n  {'='*50}")
    print(f"  Running on {dataset_name}")
    print(f"  {'='*50}")
    
    # Get dataset-specific config
    config = get_dataset_config(dataset_name)
    print(f"  Config: LR={config['learning_rate']}, Dropout={config['dropout']}, Hidden={config['hidden_dim']}")
    
    # Split
    print("  Performing scaffold split...")
    train_idx, val_idx, test_idx = scaffold_split(smiles, y, test_size=0.2, val_size=0.1, seed=42)
    
    # Store original test values
    y_test_orig = y[test_idx]
    
    # Normalize target using training statistics
    y_train_orig = y[train_idx]
    y_train_mean = y_train_orig.mean()
    y_train_std = y_train_orig.std()
    
    y_train_norm = (y[train_idx] - y_train_mean) / y_train_std
    y_val_norm = (y[val_idx] - y_train_mean) / y_train_std
    y_test_norm = (y[test_idx] - y_train_mean) / y_train_std
    
    print(f"  Target stats: mean={y_train_mean:.3f}, std={y_train_std:.3f}")
    
    # Create datasets
    print("  Creating datasets...")
    train_dataset = MoleculeDataset(smiles[train_idx], y_train_norm)
    val_dataset = MoleculeDataset(smiles[val_idx], y_val_norm)
    test_dataset = MoleculeDataset(smiles[test_idx], y_test_norm)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model with dataset-specific params
    print("  Training model...")
    model = HybridGCNModel(
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    )
    
    trainer = ModelTrainer(model, device, learning_rate=config['learning_rate'])
    
    # Train
    val_metrics, _, _ = trainer.train(train_loader, val_loader, epochs=config['epochs'], verbose=True)
    
    # Test
    test_metrics_norm, test_preds_norm, test_targets_norm = trainer.evaluate(test_loader)
    
    # Convert to original scale
    test_preds_orig = test_preds_norm * y_train_std + y_train_mean
    
    # Calculate metrics
    rmse_original = np.sqrt(mean_squared_error(y_test_orig, test_preds_orig))
    mae_original = mean_absolute_error(y_test_orig, test_preds_orig)
    r2_original = r2_score(y_test_orig, test_preds_orig)
    
    print(f"\n  Results for {dataset_name}:")
    print(f"    Original RMSE: {rmse_original:.4f}")
    print(f"    Original R2:   {r2_original:.4f}")
    
    return {
        'dataset': dataset_name,
        'n_molecules': len(smiles),
        'train_size': len(train_idx),
        'test_size': len(test_idx),
        'r2': r2_original,
        'rmse': rmse_original,
        'mae': mae_original,
        'config': config
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("OPTIMIZED MULTI-DATASET SCAFFOLD SPLIT VALIDATION")
    print("With dataset-specific hyperparameters")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load datasets
    print("\n[1/3] Loading datasets...")
    datasets = [
        load_esol(),
        load_lipophilicity(),
    ]
    
    # Evaluate
    print("\n[2/3] Running scaffold split evaluations...")
    results = []
    
    for smiles, y, name in datasets:
        result = evaluate_on_dataset(name, smiles, y, device)
        results.append(result)
    
    # Results
    print("\n[3/3] Final Results")
    print("\n" + "=" * 80)
    print("FINAL RESULTS: SCAFFOLD SPLIT VALIDATION")
    print("=" * 80)
    
    print("\n{:<15} {:>10} {:>10} {:>10} {:>12} {:>10}".format(
        'Dataset', 'Molecules', 'Train', 'Test', 'R2', 'RMSE'
    ))
    print("-" * 70)
    
    for r in results:
        print("{:<15} {:>10,} {:>10,} {:>10,} {:>12.4f} {:>10.4f}".format(
            r['dataset'], r['n_molecules'], r['train_size'], 
            r['test_size'], r['r2'], r['rmse']
        ))
    
    # Literature comparison
    print("\n" + "=" * 80)
    print("COMPARISON WITH LITERATURE (RMSE)")
    print("=" * 80)
    
    literature = {
        'ESOL': 0.877,
        'Lipophilicity': 0.721
    }
    
    for r in results:
        improvement = ((literature[r['dataset']] - r['rmse']) / literature[r['dataset']]) * 100
        status = "BETTER" if r['rmse'] < literature[r['dataset']] else "NEEDS IMPROVEMENT"
        print(f"  {r['dataset']}: Ours={r['rmse']:.3f} vs Lit={literature[r['dataset']]:.3f} -> {status} ({improvement:+.1f}%)")
    
    # Save
    import json
    with open('optimized_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n[+] Results saved to: optimized_results.json")
    
    # LaTeX table
    print("\n" + "=" * 80)
    print("LATEX TABLE FOR PAPER")
    print("=" * 80)
    
    latex_table = r"\begin{table}[htbp]" + "\n"
    latex_table += r"\centering" + "\n"
    latex_table += r"\caption{Scaffold split validation results}" + "\n"
    latex_table += r"\begin{tabular}{lcccc}" + "\n"
    latex_table += r"\hline" + "\n"
    latex_table += r"Dataset & \# Molecules & R$^2$ & RMSE & MAE \\" + "\n"
    latex_table += r"\hline" + "\n"
    
    for r in results:
        latex_table += f"{r['dataset']} & {r['n_molecules']} & {r['r2']:.3f} & {r['rmse']:.3f} & {r['mae']:.3f} \\\\\n"
    
    latex_table += r"\hline" + "\n"
    latex_table += r"\end{tabular}" + "\n"
    latex_table += r"\label{tab:scaffold_results}" + "\n"
    latex_table += r"\end{table}" + "\n"
    
    print(latex_table)
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = main()