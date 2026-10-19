"""
Scaffold Split Implementation for Molecular Property Prediction
Essential for Q1 methodological rigor - tests generalization to novel chemical scaffolds

Run: python scaffold_split_pipeline.py
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

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ============================================================================
# SCAFFOLD GENERATION AND SPLITTING
# ============================================================================

def generate_scaffold(smiles):
    """
    Generate Murcko scaffold from SMILES string.
    Murcko scaffolds represent the core molecular framework.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Generate Murcko scaffold
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return scaffold
    except Exception as e:
        return None

def scaffold_split(smiles_list, y_values, test_size=0.2, val_size=0.1, seed=42):
    """
    Split dataset by molecular scaffolds.
    
    This ensures that molecules with similar core structures are kept together,
    testing the model's ability to generalize to truly novel chemical space.
    
    Args:
        smiles_list: List of SMILES strings
        y_values: List of target values
        test_size: Proportion for test set (default: 0.2)
        val_size: Proportion for validation set (default: 0.1)
        seed: Random seed for reproducibility
    
    Returns:
        train_idx, val_idx, test_idx: Indices for each split
    """
    random.seed(seed)
    
    # Generate scaffolds for each molecule
    print("  Generating scaffolds for all molecules...")
    scaffold_to_indices = defaultdict(list)
    no_scaffold_count = 0
    
    for idx, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles)
        if scaffold is not None:
            scaffold_to_indices[scaffold].append(idx)
        else:
            no_scaffold_count += 1
            # Put molecules without scaffolds into their own group
            scaffold_to_indices[f"NO_SCAFFOLD_{idx}"].append(idx)
    
    if no_scaffold_count > 0:
        print(f"  Warning: {no_scaffold_count} molecules could not generate scaffolds")
    
    # Get unique scaffolds
    scaffolds = list(scaffold_to_indices.keys())
    print(f"  Total unique scaffolds: {len(scaffolds)}")
    
    # Shuffle scaffolds
    random.shuffle(scaffolds)
    
    # Calculate split sizes
    n_test = max(1, int(len(scaffolds) * test_size))
    n_val = max(1, int(len(scaffolds) * val_size))
    
    # Split scaffolds
    test_scaffolds = set(scaffolds[:n_test])
    val_scaffolds = set(scaffolds[n_test:n_test + n_val])
    train_scaffolds = set(scaffolds[n_test + n_val:])
    
    # Map back to molecule indices
    train_idx = []
    val_idx = []
    test_idx = []
    
    for scaffold in train_scaffolds:
        train_idx.extend(scaffold_to_indices[scaffold])
    
    for scaffold in val_scaffolds:
        val_idx.extend(scaffold_to_indices[scaffold])
    
    for scaffold in test_scaffolds:
        test_idx.extend(scaffold_to_indices[scaffold])
    
    # Shuffle indices within each split (optional but good practice)
    random.shuffle(train_idx)
    random.shuffle(val_idx)
    random.shuffle(test_idx)
    
    print(f"  Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    print(f"  Unique scaffolds: Train={len(train_scaffolds)}, Val={len(val_scaffolds)}, Test={len(test_scaffolds)}")
    
    return train_idx, val_idx, test_idx

def random_split(smiles_list, y_values, test_size=0.2, val_size=0.1, seed=42):
    """
    Baseline random split for comparison.
    """
    from sklearn.model_selection import train_test_split
    
    np.random.seed(seed)
    indices = np.arange(len(smiles_list))
    
    # First split: separate test set
    train_val_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed)
    
    # Second split: separate validation from train
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size/(1-test_size), random_state=seed)
    
    print(f"  Random split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    return train_idx, val_idx, test_idx


# ============================================================================
# DATA PROCESSING (Same as before)
# ============================================================================

class MolecularGraphBuilder:
    """Convert SMILES to PyTorch Geometric graph"""
    
    @staticmethod
    def get_atom_features(atom, mol):
        features = []
        # Atomic number
        features.append(min(atom.GetAtomicNum() / 100.0, 1.0))
        # Degree
        features.append(min(atom.GetDegree(), 5) / 5.0)
        # Formal charge
        features.append((atom.GetFormalCharge() + 1) / 3.0)
        # Number of hydrogens
        features.append(min(atom.GetTotalNumHs(), 4) / 4.0)
        # Hybridization
        hyb = atom.GetHybridization()
        features.append(1.0 if hyb == Chem.rdchem.HybridizationType.SP else 0.0)
        features.append(1.0 if hyb == Chem.rdchem.HybridizationType.SP2 else 0.0)
        features.append(1.0 if hyb == Chem.rdchem.HybridizationType.SP3 else 0.0)
        # Aromaticity
        features.append(1.0 if atom.GetIsAromatic() else 0.0)
        # Ring membership
        is_in_ring = atom.IsInRing()
        features.append(1.0 if is_in_ring else 0.0)
        # Ring size
        if is_in_ring:
            ring_info = mol.GetRingInfo()
            ring_sizes = ring_info.AtomRingSizes(atom.GetIdx())
            features.append(min(ring_sizes[0], 8) / 8.0 if ring_sizes else 0.0)
        else:
            features.append(0.0)
        # Chirality
        chiral = atom.GetChiralTag()
        features.append(1.0 if chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW else 0.0)
        features.append(1.0 if chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW else 0.0)
        # Mass
        features.append(min(atom.GetMass(), 200) / 200.0)
        # Pad to 20 features
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


class MolecularDescriptorExtractor:
    """Extract physicochemical descriptors"""
    
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


class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, y_values, descriptor_scaler=None):
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
        
        if descriptor_scaler is not None:
            self.valid_descriptors = descriptor_scaler.transform(self.valid_descriptors)
        else:
            mean = self.valid_descriptors.mean(axis=0)
            std = self.valid_descriptors.std(axis=0)
            self.valid_descriptors = (self.valid_descriptors - mean) / (std + 1e-8)
    
    def len(self):
        return len(self.valid_graphs)
    
    def get(self, idx):
        return self.valid_graphs[idx], torch.tensor(self.valid_descriptors[idx], dtype=torch.float), self.valid_y[idx]


# ============================================================================
# MODEL (Simplified Hybrid GCN)
# ============================================================================

class HybridGCNModel(nn.Module):
    def __init__(self, node_features_dim=20, hidden_dim=128, descriptor_dim=50, 
                 num_classes=1, dropout=0.2):
        super().__init__()
        
        self.conv1 = GCNConv(node_features_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
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
        
        x = F.relu(self.conv3(x, edge_index))
        
        gnn_mean = global_mean_pool(x, batch)
        gnn_max = global_max_pool(x, batch)
        gnn_features = torch.cat([gnn_mean, gnn_max], dim=1)
        
        desc_features = self.descriptor_branch(descriptors)
        
        combined = torch.cat([gnn_features, desc_features], dim=1)
        combined = self.fusion(combined)
        
        return self.predictor(combined)


# ============================================================================
# TRAINER
# ============================================================================

class ModelTrainer:
    def __init__(self, model, device, learning_rate=1e-3, weight_decay=1e-5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
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
        }
    
    def train(self, train_loader, val_loader, epochs=100, verbose=True):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            self.scheduler.step(val_metrics['loss'])
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                self.patience_counter += 1
                if self.patience_counter >= 20:
                    break
            
            if verbose and epoch % 20 == 0:
                print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val R2: {val_metrics['r2']:.4f}")
        
        self.model.load_state_dict(torch.load('best_model.pt'))
        return self.evaluate(val_loader)


# ============================================================================
# MAIN SCAFFOLD SPLIT EXPERIMENT
# ============================================================================

def run_scaffold_split_experiment():
    """Run comparison between random split and scaffold split"""
    
    print("=" * 80)
    print("SCAFFOLD SPLIT VALIDATION STUDY")
    print("Testing generalization to novel molecular scaffolds")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    print("\n[1/4] Loading ESOL dataset...")
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv'
    data = pd.read_csv(url)
    smiles = data['smiles'].values
    y = data['measured log solubility in mols per litre'].values
    
    # Normalize target
    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / y_std
    
    print(f"Total molecules: {len(smiles)}")
    
    # ========================================================================
    # EXPERIMENT 1: Scaffold Split (Rigorous)
    # ========================================================================
    
    print("\n[2/4] Running Scaffold Split (Rigorous)...")
    print("-" * 40)
    
    train_idx_scaffold, val_idx_scaffold, test_idx_scaffold = scaffold_split(
        smiles, y_norm, test_size=0.2, val_size=0.1, seed=42
    )
    
    # Create datasets with scaffold split
    print("\n  Creating datasets...")
    train_dataset_scaffold = MoleculeDataset(smiles[train_idx_scaffold], y_norm[train_idx_scaffold])
    val_dataset_scaffold = MoleculeDataset(smiles[val_idx_scaffold], y_norm[val_idx_scaffold])
    test_dataset_scaffold = MoleculeDataset(smiles[test_idx_scaffold], y_norm[test_idx_scaffold])
    
    train_loader_scaffold = DataLoader(train_dataset_scaffold, batch_size=32, shuffle=True)
    val_loader_scaffold = DataLoader(val_dataset_scaffold, batch_size=32, shuffle=False)
    test_loader_scaffold = DataLoader(test_dataset_scaffold, batch_size=32, shuffle=False)
    
    # Train model with scaffold split
    print("\n  Training model...")
    model_scaffold = HybridGCNModel(node_features_dim=20, hidden_dim=128, descriptor_dim=50, dropout=0.2)
    trainer_scaffold = ModelTrainer(model_scaffold, device, learning_rate=0.001)
    
    val_metrics_scaffold = trainer_scaffold.train(train_loader_scaffold, val_loader_scaffold, epochs=100, verbose=True)
    test_metrics_scaffold = trainer_scaffold.evaluate(test_loader_scaffold)
    
    # ========================================================================
    # EXPERIMENT 2: Random Split (Comparison Baseline)
    # ========================================================================
    
    print("\n[3/4] Running Random Split (Comparison Baseline)...")
    print("-" * 40)
    
    train_idx_random, val_idx_random, test_idx_random = random_split(
        smiles, y_norm, test_size=0.2, val_size=0.1, seed=42
    )
    
    print("\n  Creating datasets...")
    train_dataset_random = MoleculeDataset(smiles[train_idx_random], y_norm[train_idx_random])
    val_dataset_random = MoleculeDataset(smiles[val_idx_random], y_norm[val_idx_random])
    test_dataset_random = MoleculeDataset(smiles[test_idx_random], y_norm[test_idx_random])
    
    train_loader_random = DataLoader(train_dataset_random, batch_size=32, shuffle=True)
    val_loader_random = DataLoader(val_dataset_random, batch_size=32, shuffle=False)
    test_loader_random = DataLoader(test_dataset_random, batch_size=32, shuffle=False)
    
    print("\n  Training model...")
    model_random = HybridGCNModel(node_features_dim=20, hidden_dim=128, descriptor_dim=50, dropout=0.2)
    trainer_random = ModelTrainer(model_random, device, learning_rate=0.001)
    
    val_metrics_random = trainer_random.train(train_loader_random, val_loader_random, epochs=100, verbose=True)
    test_metrics_random = trainer_random.evaluate(test_loader_random)
    
    # ========================================================================
    # RESULTS COMPARISON
    # ========================================================================
    
    print("\n[4/4] Results Comparison")
    print("=" * 80)
    print("SPLIT METHOD COMPARISON")
    print("=" * 80)
    
    print("\n{:<20} {:>12} {:>12} {:>12}".format(
        'Split Method', 'Test R2', 'Test RMSE', 'Test MAE'
    ))
    print("-" * 60)
    print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f}".format(
        'Random Split', 
        test_metrics_random['r2'], 
        test_metrics_random['rmse'],
        test_metrics_random['mae']
    ))
    print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f}".format(
        'Scaffold Split', 
        test_metrics_scaffold['r2'], 
        test_metrics_scaffold['rmse'],
        test_metrics_scaffold['mae']
    ))
    
    # Calculate performance gap
    performance_gap = test_metrics_random['r2'] - test_metrics_scaffold['r2']
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print(f"Random Split R2:      {test_metrics_random['r2']:.4f}")
    print(f"Scaffold Split R2:    {test_metrics_scaffold['r2']:.4f}")
    print(f"Performance Gap:      {performance_gap:.4f} ({performance_gap*100:.1f}% lower)")
    
    if performance_gap > 0:
        print("\nInterpretation: Random split overestimates performance by {:.1f}%".format(performance_gap*100))
        print("Scaffold split provides more realistic estimate of generalization to novel molecules.")
    else:
        print("\nInterpretation: Model generalizes well to novel scaffolds!")
    
    # Save results
    results = {
        'random_split': {
            'r2': test_metrics_random['r2'],
            'rmse': test_metrics_random['rmse'],
            'mae': test_metrics_random['mae']
        },
        'scaffold_split': {
            'r2': test_metrics_scaffold['r2'],
            'rmse': test_metrics_scaffold['rmse'],
            'mae': test_metrics_scaffold['mae']
        },
        'performance_gap': performance_gap
    }
    
    import json
    with open('scaffold_split_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n[+] Results saved to: scaffold_split_results.json")
    
    # Generate LaTeX table for paper
    print("\n" + "=" * 80)
    print("LATEX TABLE FOR PAPER")
    print("=" * 80)
    
    latex_table = r"\begin{table}[htbp]" + "\n"
    latex_table += r"\centering" + "\n"
    latex_table += r"\caption{Comparison of random vs. scaffold split validation}" + "\n"
    latex_table += r"\begin{tabular}{lccc}" + "\n"
    latex_table += r"\hline" + "\n"
    latex_table += r"Split Method & R$^2$ & RMSE & MAE \\" + "\n"
    latex_table += r"\hline" + "\n"
    latex_table += f"Random Split & {test_metrics_random['r2']:.3f} & {test_metrics_random['rmse']:.3f} & {test_metrics_random['mae']:.3f} \\\\\n"
    latex_table += f"Scaffold Split & {test_metrics_scaffold['r2']:.3f} & {test_metrics_scaffold['rmse']:.3f} & {test_metrics_scaffold['mae']:.3f} \\\\\n"
    latex_table += r"\hline" + "\n"
    latex_table += r"\end{tabular}" + "\n"
    latex_table += r"\label{tab:split_comparison}" + "\n"
    latex_table += r"\end{table}" + "\n"
    
    print(latex_table)
    
    with open('split_comparison_table.tex', 'w', encoding='utf-8') as f:
        f.write(latex_table)
    
    print("[+] LaTeX table saved to: split_comparison_table.tex")
    
    print("\n" + "=" * 80)
    print("SCAFFOLD SPLIT STUDY COMPLETE")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = run_scaffold_split_experiment()