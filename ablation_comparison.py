"""
Ablation Study for Hybrid GNN + Descriptor Model 
Validates contribution of each component for Q1 publication

Run: python ablation_study.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import pandas as pd
import numpy as np
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ============================================================================
# DATA PROCESSING (Same as original pipeline)
# ============================================================================

class MolecularGraphBuilder:
    """Convert SMILES to PyTorch Geometric graph"""
    
    @staticmethod
    def get_atom_features(atom, mol):
        features = []
        # Atomic number (normalized)
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


class SimpleDataset(Dataset):
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
# MODEL VARIANTS FOR ABLATION STUDY
# ============================================================================

class HybridGCNModel(nn.Module):
    """Full model: GCN + Descriptors"""
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


class HybridGATModel(nn.Module):
    """Full model: GAT + Descriptors"""
    def __init__(self, node_features_dim=20, hidden_dim=128, descriptor_dim=50, 
                 num_classes=1, dropout=0.2):
        super().__init__()
        
        self.conv1 = GATConv(node_features_dim, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim * 4)
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
        
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.batch_norm1(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.batch_norm2(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        
        gnn_mean = global_mean_pool(x, batch)
        gnn_max = global_max_pool(x, batch)
        gnn_features = torch.cat([gnn_mean, gnn_max], dim=1)
        
        desc_features = self.descriptor_branch(descriptors)
        
        combined = torch.cat([gnn_features, desc_features], dim=1)
        combined = self.fusion(combined)
        
        return self.predictor(combined)


class GNNOnlyModel(nn.Module):
    """Ablation: GCN only (no descriptors)"""
    def __init__(self, node_features_dim=20, hidden_dim=128, num_classes=1, dropout=0.2):
        super().__init__()
        
        self.conv1 = GCNConv(node_features_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, graph_data, descriptors=None):
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
        
        return self.predictor(gnn_features)


class DescriptorOnlyModel(nn.Module):
    """Ablation: Descriptors only (no GNN)"""
    def __init__(self, descriptor_dim=50, num_classes=1, dropout=0.2, **kwargs):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(descriptor_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, graph_data, descriptors):
        return self.mlp(descriptors)


class ShallowHybridModel(nn.Module):
    """Ablation: Hybrid with only 1 GNN layer"""
    def __init__(self, node_features_dim=20, hidden_dim=128, descriptor_dim=50, 
                 num_classes=1, dropout=0.2):
        super().__init__()
        
        self.conv1 = GCNConv(node_features_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
        self.descriptor_branch = nn.Sequential(
            nn.Linear(descriptor_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(dropout)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, num_classes)
        )
    
    def forward(self, graph_data, descriptors):
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        
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
# ABLATION STUDY
# ============================================================================

def run_ablation_study():
    """Run complete ablation study and generate report"""
    
    print("=" * 80)
    print("ABLATION STUDY: Hybrid GNN + Descriptor Model")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    print("\n[1/5] Loading ESOL dataset...")
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv'
    data = pd.read_csv(url)
    smiles = data['smiles'].values
    y = data['measured log solubility in mols per litre'].values
    
    # Normalize target
    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / y_std
    
    # Split data
    train_idx, temp_idx = train_test_split(np.arange(len(smiles)), test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Create datasets
    print("\n[2/5] Creating datasets...")
    train_dataset = SimpleDataset(smiles[train_idx], y_norm[train_idx])
    val_dataset = SimpleDataset(smiles[val_idx], y_norm[val_idx])
    test_dataset = SimpleDataset(smiles[test_idx], y_norm[test_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Define models to test
    models_to_test = [
        {
            'name': 'Full Hybrid (GCN + Descriptors)',
            'model_class': HybridGCNModel,
            'params': {'descriptor_dim': 50, 'hidden_dim': 128, 'dropout': 0.2},
        },
        {
            'name': 'GNN Only (No Descriptors)',
            'model_class': GNNOnlyModel,
            'params': {'hidden_dim': 128, 'dropout': 0.2},
        },
        {
            'name': 'Descriptors Only (No GNN)',
            'model_class': DescriptorOnlyModel,
            'params': {'descriptor_dim': 50, 'dropout': 0.2},
        },
        {
            'name': 'Hybrid with GAT (vs GCN)',
            'model_class': HybridGATModel,
            'params': {'descriptor_dim': 50, 'hidden_dim': 128, 'dropout': 0.2},
        },
        {
            'name': 'Shallow Hybrid (1 GNN Layer)',
            'model_class': ShallowHybridModel,
            'params': {'descriptor_dim': 50, 'hidden_dim': 128, 'dropout': 0.2},
        }
    ]
    
    results = []
    
    print("\n[3/5] Running ablation experiments...")
    print("-" * 80)
    
    for i, model_config in enumerate(models_to_test):
        print(f"\nExperiment {i+1}/{len(models_to_test)}: {model_config['name']}")
        
        start_time = time.time()
        
        model = model_config['model_class'](
            node_features_dim=20,
            num_classes=1,
            **model_config['params']
        )
        
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {n_params:,}")
        
        trainer = ModelTrainer(model, device, learning_rate=0.001)
        val_metrics = trainer.train(train_loader, val_loader, epochs=100, verbose=True)
        test_metrics = trainer.evaluate(test_loader)
        
        elapsed_time = time.time() - start_time
        
        result = {
            'model': model_config['name'],
            'n_params': n_params,
            'val_r2': val_metrics['r2'],
            'val_rmse': val_metrics['rmse'],
            'test_r2': test_metrics['r2'],
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae'],
            'time_seconds': elapsed_time
        }
        results.append(result)
        
        print(f"  >> Test R2: {test_metrics['r2']:.4f} | Time: {elapsed_time:.1f}s")
    
    # Results DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n[4/5] Generating ablation report...")
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)
    
    print("\n{:<35} {:>12} {:>12} {:>12} {:>10}".format(
        'Model', 'Test R2', 'Test RMSE', 'Test MAE', 'Params'
    ))
    print("-" * 85)
    
    for _, row in results_df.iterrows():
        print("{:<35} {:>12.4f} {:>12.4f} {:>12.4f} {:>10,}".format(
            row['model'][:35], row['test_r2'], row['test_rmse'], row['test_mae'], row['n_params']
        ))
    
    # Analysis
    baseline_r2 = results_df[results_df['model'] == 'GNN Only (No Descriptors)']['test_r2'].values[0]
    full_r2 = results_df[results_df['model'] == 'Full Hybrid (GCN + Descriptors)']['test_r2'].values[0]
    improvement = (full_r2 - baseline_r2) * 100
    
    print("\n" + "=" * 80)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 80)
    print(f"Baseline (GNN Only) R2:     {baseline_r2:.4f}")
    print(f"Full Hybrid Model R2:       {full_r2:.4f}")
    print(f"Absolute Improvement:       {full_r2 - baseline_r2:.4f}")
    print(f"Relative Improvement:       {improvement:.1f}%")
    
    # Conclusions
    print("\n" + "=" * 80)
    print("CONCLUSIONS FOR Q1 PAPER")
    print("=" * 80)
    
    conclusions = []
    
    if full_r2 > baseline_r2 + 0.05:
        conclusions.append(f"[+] The hybrid model significantly outperforms GNN-only baseline (+{improvement:.1f}% R2 improvement)")
    
    desc_r2 = results_df[results_df['model'] == 'Descriptors Only (No GNN)']['test_r2'].values[0]
    if full_r2 > desc_r2 + 0.05:
        conclusions.append(f"[+] GNN captures complementary structural information beyond descriptors (desc-only R2={desc_r2:.3f})")
    
    gat_r2 = results_df[results_df['model'] == 'Hybrid with GAT (vs GCN)']['test_r2'].values[0]
    if gat_r2 > full_r2:
        conclusions.append(f"[+] GAT attention mechanism outperforms GCN (+{(gat_r2 - full_r2)*100:.1f}% R2)")
    else:
        conclusions.append(f"[-] GCN performs better than GAT for this task (+{(full_r2 - gat_r2)*100:.1f}% R2)")
    
    shallow_r2 = results_df[results_df['model'] == 'Shallow Hybrid (1 GNN Layer)']['test_r2'].values[0]
    if full_r2 > shallow_r2:
        conclusions.append(f"[+] Deeper GNN (3 layers) captures more complex patterns (+{(full_r2 - shallow_r2)*100:.1f}% R2)")
    
    for conclusion in conclusions:
        print(conclusion)
    
    # Save results
    print("\n[5/5] Saving results...")
    
    results_df.to_csv('ablation_study_results.csv', index=False)
    print("[+] Results saved to: ablation_study_results.csv")
    
    with open('ablation_study_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ABLATION STUDY REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write(results_df.to_string() + "\n\n")
        f.write("KEY FINDINGS:\n")
        for conclusion in conclusions:
            f.write(conclusion + "\n")
    
    print("[+] Report saved to: ablation_study_report.txt")
    
    # LaTeX table
    print("\n" + "=" * 80)
    print("LATEX TABLE FOR PAPER")
    print("=" * 80)
    
    latex_table = r"\begin{table}[htbp]" + "\n"
    latex_table += r"\centering" + "\n"
    latex_table += r"\caption{Ablation study results on ESOL dataset}" + "\n"
    latex_table += r"\begin{tabular}{lcccc}" + "\n"
    latex_table += r"\hline" + "\n"
    latex_table += r"Model & R$^2$ & RMSE & MAE & \# Params \\" + "\n"
    latex_table += r"\hline" + "\n"
    
    for _, row in results_df.iterrows():
        latex_table += f"{row['model']} & {row['test_r2']:.3f} & {row['test_rmse']:.3f} & {row['test_mae']:.3f} & {row['n_params']:,} \\\\\n"
    
    latex_table += r"\hline" + "\n"
    latex_table += r"\end{tabular}" + "\n"
    latex_table += r"\label{tab:ablation}" + "\n"
    latex_table += r"\end{table}" + "\n"
    
    print(latex_table)
    
    with open('ablation_table.tex', 'w', encoding='utf-8') as f:
        f.write(latex_table)
    print("[+] LaTeX table saved to: ablation_table.tex")
    
    print("\n" + "=" * 80)
    print("ABLATION STUDY COMPLETE")
    print("=" * 80)
    
    return results_df


if __name__ == "__main__":
    results = run_ablation_study()