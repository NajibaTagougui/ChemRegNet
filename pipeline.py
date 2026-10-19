"""
Hybrid Descriptor + Graph Neural Network for Molecular Property Prediction
Complete Pipeline for Q1 Publication-Ready Research (FIXED VERSION)

Author: Research Pipeline
Requirements: pip install torch torch-geometric rdkit pandas numpy scikit-learn optuna
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('rdkit').setLevel(logging.ERROR)

# Also suppress RDKit deprecation warnings
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
# PART 1: MOLECULAR GRAPH CONSTRUCTION (FIXED)
# ============================================================================

class MolecularGraphBuilder:
    """Convert SMILES to PyTorch Geometric graph with rich atom/bond features"""
    
    @staticmethod
    def get_atom_features(atom, mol):
        """Extract 20-dimensional atom feature vector"""
        features = []
        
        # Atomic number (normalized)
        atomic_num = atom.GetAtomicNum()
        features.append(min(atomic_num / 100.0, 1.0))
        
        # Degree (number of explicit bonds)
        features.append(min(atom.GetDegree(), 5) / 5.0)
        
        # Formal charge (normalized)
        features.append((atom.GetFormalCharge() + 1) / 3.0)
        
        # Number of hydrogens (normalized)
        features.append(min(atom.GetTotalNumHs(), 4) / 4.0)
        
        # Hybridization (one-hot)
        hyb = atom.GetHybridization()
        features.append(1.0 if hyb == Chem.rdchem.HybridizationType.SP else 0.0)
        features.append(1.0 if hyb == Chem.rdchem.HybridizationType.SP2 else 0.0)
        features.append(1.0 if hyb == Chem.rdchem.HybridizationType.SP3 else 0.0)
        
        # Aromaticity
        features.append(1.0 if atom.GetIsAromatic() else 0.0)
        
        # Ring membership
        is_in_ring = atom.IsInRing()
        features.append(1.0 if is_in_ring else 0.0)
        
        # Ring size (if in ring)
        if is_in_ring:
            # Get ring information from the molecule
            ring_info = mol.GetRingInfo()
            ring_sizes = ring_info.AtomRingSizes(atom.GetIdx())
            if ring_sizes:
                features.append(min(ring_sizes[0], 8) / 8.0)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Chirality
        chiral = atom.GetChiralTag()
        features.append(1.0 if chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW else 0.0)
        features.append(1.0 if chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW else 0.0)
        
        # Mass (normalized)
        features.append(min(atom.GetMass(), 200) / 200.0)
        
        # Explicit valence (normalized)
        features.append(min(atom.GetExplicitValence(), 6) / 6.0)
        
        # Implicit valence (normalized)
        features.append(min(atom.GetImplicitValence(), 4) / 4.0)
        
        # Number of radical electrons (normalized)
        features.append(min(atom.GetNumRadicalElectrons(), 2) / 2.0)
        
        # Is in ring of size 3,4,5,6,7,8
        if is_in_ring:
            ring_info = mol.GetRingInfo()
            ring_sizes = ring_info.AtomRingSizes(atom.GetIdx())
            for size in [3, 4, 5, 6, 7, 8]:
                features.append(1.0 if size in ring_sizes else 0.0)
        else:
            features.extend([0.0] * 6)
        
        # Ensure we have exactly 20 features (pad if needed)
        while len(features) < 20:
            features.append(0.0)
        features = features[:20]
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def get_bond_features(bond):
        """Extract bond feature vector"""
        features = []
        
        # Bond type (one-hot)
        bond_type = bond.GetBondType()
        features.append(1.0 if bond_type == Chem.rdchem.BondType.SINGLE else 0.0)
        features.append(1.0 if bond_type == Chem.rdchem.BondType.DOUBLE else 0.0)
        features.append(1.0 if bond_type == Chem.rdchem.BondType.TRIPLE else 0.0)
        features.append(1.0 if bond_type == Chem.rdchem.BondType.AROMATIC else 0.0)
        
        # Conjugation
        features.append(1.0 if bond.GetIsConjugated() else 0.0)
        
        # In ring
        features.append(1.0 if bond.IsInRing() else 0.0)
        
        # Stereo
        stereo = bond.GetStereo()
        features.append(1.0 if stereo == Chem.rdchem.BondStereo.STEREOANY else 0.0)
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def mol_to_graph(smiles):
        """Convert SMILES string to PyTorch Geometric Data object"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Sanitize the molecule (important for valid chemistry)
            Chem.SanitizeMol(mol)
            
            # Don't add hydrogens - it can cause issues with ring detection
            # Keep the molecule as is for graph construction
            
            # Node features
            node_features = []
            for atom in mol.GetAtoms():
                node_features.append(MolecularGraphBuilder.get_atom_features(atom, mol))
            
            # Edge indices and features
            edge_indices = []
            edge_features = []
            
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                # Add both directions for undirected graph
                edge_indices.append([i, j])
                edge_indices.append([j, i])
                
                bond_feat = MolecularGraphBuilder.get_bond_features(bond)
                edge_features.append(bond_feat)
                edge_features.append(bond_feat)
            
            if len(edge_indices) == 0:
                # Handle single-atom molecules
                edge_indices = [[0, 0]]
                edge_features = [np.zeros(7, dtype=np.float32)]
            
            # Convert to tensors
            x = torch.tensor(np.array(node_features), dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float)
            
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
        except Exception as e:
            # print(f"Error processing SMILES: {smiles[:50]}... Error: {str(e)[:100]}")
            return None


# ============================================================================
# PART 2: MOLECULAR DESCRIPTOR EXTRACTION (FIXED)
# ============================================================================

class MolecularDescriptorExtractor:
    """Extract physicochemical descriptors for MLP branch"""
    
    @staticmethod
    def compute_descriptors(smiles):
        """Compute key molecular descriptors safely"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(50, dtype=np.float32)
            
            # Sanitize
            Chem.SanitizeMol(mol)
            
            descriptors = []
            
            # Basic 1D descriptors
            descriptors.append(Descriptors.MolWt(mol))                    # Molecular weight
            descriptors.append(Descriptors.MolLogP(mol))                  # LogP
            descriptors.append(Descriptors.NumHAcceptors(mol))            # H-bond acceptors
            descriptors.append(Descriptors.NumHDonors(mol))               # H-bond donors
            descriptors.append(Descriptors.TPSA(mol))                     # Polar surface area
            descriptors.append(Descriptors.NumRotatableBonds(mol))        # Rotatable bonds
            descriptors.append(Descriptors.NumAromaticRings(mol))         # Aromatic rings
            descriptors.append(Descriptors.NumAliphaticRings(mol))        # Aliphatic rings
            descriptors.append(Descriptors.NumSaturatedRings(mol))        # Saturated rings
            descriptors.append(Descriptors.NumHeteroatoms(mol))           # Heteroatoms
            
            # More complex descriptors (using correct capitalization)
            descriptors.append(Descriptors.FractionCSP3(mol))             # sp3 fraction
            descriptors.append(Descriptors.BalabanJ(mol))                 # Balaban's J index
            descriptors.append(Descriptors.BertzCT(mol))                  # Bertz CT index
            descriptors.append(Descriptors.HallKierAlpha(mol))            # Hall-Kier alpha
            descriptors.append(Descriptors.Kappa1(mol))                   # Kappa shape index 1
            descriptors.append(Descriptors.Kappa2(mol))                   # Kappa shape 2
            descriptors.append(Descriptors.Kappa3(mol))                   # Kappa shape 3
            
            # E-state and connectivity
            descriptors.append(Descriptors.HeavyAtomCount(mol))           # Heavy atoms
            descriptors.append(Descriptors.HeavyAtomMolWt(mol))           # Heavy atom MW
            descriptors.append(Descriptors.NHOHCount(mol))                # NHOH count
            descriptors.append(Descriptors.NOCount(mol))                  # NO count
            
            # Additional safe descriptors
            descriptors.append(rdMolDescriptors.CalcTPSA(mol))            # Topological PSA
            descriptors.append(Descriptors.MolMR(mol))                    # Molar refractivity
            descriptors.append(Descriptors.ExactMolWt(mol))               # Exact mass
            descriptors.append(Descriptors.NumValenceElectrons(mol))      # Valence electrons
            descriptors.append(Descriptors.NumRadicalElectrons(mol))      # Radical electrons
            
            # Charge descriptors
            descriptors.append(Descriptors.MaxAbsPartialCharge(mol) if Descriptors.MaxAbsPartialCharge(mol) else 0.0)
            descriptors.append(Descriptors.MinAbsPartialCharge(mol) if Descriptors.MinAbsPartialCharge(mol) else 0.0)
            descriptors.append(Descriptors.MaxPartialCharge(mol) if Descriptors.MaxPartialCharge(mol) else 0.0)
            descriptors.append(Descriptors.MinPartialCharge(mol) if Descriptors.MinPartialCharge(mol) else 0.0)
            
            # E-state indices
            descriptors.append(Descriptors.MaxEStateIndex(mol) if Descriptors.MaxEStateIndex(mol) else 0.0)
            descriptors.append(Descriptors.MinEStateIndex(mol) if Descriptors.MinEStateIndex(mol) else 0.0)
            descriptors.append(Descriptors.MaxAbsEStateIndex(mol) if Descriptors.MaxAbsEStateIndex(mol) else 0.0)
            descriptors.append(Descriptors.MinAbsEStateIndex(mol) if Descriptors.MinAbsEStateIndex(mol) else 0.0)
            
            # Ring descriptors
            ring_info = mol.GetRingInfo()
            descriptors.append(len(ring_info.AtomRings()))                # Total rings
            
            # Additional descriptors to reach 50
            descriptors.append(Descriptors.NumAliphaticRings(mol))
            descriptors.append(Descriptors.NumAromaticRings(mol))
            descriptors.append(Descriptors.NumSaturatedRings(mol))
            descriptors.append(Descriptors.Chi0n(mol))                    # Chi0n
            descriptors.append(Descriptors.Chi1n(mol))                    # Chi1n
            descriptors.append(Descriptors.Chi2n(mol))                    # Chi2n
            descriptors.append(Descriptors.Chi3n(mol))                    # Chi3n
            descriptors.append(Descriptors.Chi0v(mol))                    # Chi0v
            descriptors.append(Descriptors.Chi1v(mol))                    # Chi1v
            descriptors.append(Descriptors.Chi2v(mol))                    # Chi2v
            descriptors.append(Descriptors.Chi3v(mol))                    # Chi3v
            
            # Fill to 50
            while len(descriptors) < 50:
                descriptors.append(0.0)
            
            # Trim to exactly 50
            if len(descriptors) > 50:
                descriptors = descriptors[:50]
            
            # Convert to numpy and handle NaN
            desc_array = np.array(descriptors, dtype=np.float32)
            desc_array = np.nan_to_num(desc_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            return desc_array
            
        except Exception as e:
            # Return zeros on error
            return np.zeros(50, dtype=np.float32)


# ============================================================================
# PART 3: HYBRID DATASET (SIMPLIFIED)
# ============================================================================

class SimpleDataset(Dataset):
    """Simplified dataset that handles errors gracefully"""
    
    def __init__(self, smiles_list, y_values, descriptor_scaler=None):
        super().__init__()
        
        self.valid_graphs = []
        self.valid_descriptors = []
        self.valid_y = []
        
        print(f"Processing {len(smiles_list)} molecules...")
        
        for i, (smiles, y) in enumerate(zip(smiles_list, y_values)):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{len(smiles_list)}")
            
            # Build graph
            graph = MolecularGraphBuilder.mol_to_graph(smiles)
            if graph is None:
                continue
            
            # Compute descriptors
            desc = MolecularDescriptorExtractor.compute_descriptors(smiles)
            
            self.valid_graphs.append(graph)
            self.valid_descriptors.append(desc)
            self.valid_y.append(y)
        
        # Convert to numpy
        self.valid_descriptors = np.array(self.valid_descriptors)
        self.valid_y = torch.tensor(self.valid_y, dtype=torch.float).view(-1, 1)
        
        # Scale descriptors
        if descriptor_scaler is not None:
            self.valid_descriptors = descriptor_scaler.transform(self.valid_descriptors)
        else:
            mean = self.valid_descriptors.mean(axis=0)
            std = self.valid_descriptors.std(axis=0)
            self.valid_descriptors = (self.valid_descriptors - mean) / (std + 1e-8)
        
        print(f"  Valid molecules: {len(self.valid_graphs)}/{len(smiles_list)}")
    
    def len(self):
        return len(self.valid_graphs)
    
    def get(self, idx):
        return self.valid_graphs[idx], torch.tensor(self.valid_descriptors[idx], dtype=torch.float), self.valid_y[idx]


# ============================================================================
# PART 4: HYBRID GNN MODEL
# ============================================================================

class HybridGNNModel(nn.Module):
    """Hybrid model combining GNN with molecular descriptors"""
    
    def __init__(self, node_features_dim=20, hidden_dim=128,
                 descriptor_dim=50, num_classes=1, dropout=0.2, gnn_type='GCN'):
        super().__init__()
        
        # GNN layers
        if gnn_type == 'GCN':
            self.conv1 = GCNConv(node_features_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
        else:  # GAT
            self.conv1 = GATConv(node_features_dim, hidden_dim, heads=4, concat=True)
            self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)
            self.conv3 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        
        # Descriptor MLP
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
        
        # Fusion and prediction
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
        
        # GNN branch
        x = F.relu(self.conv1(x, edge_index))
        x = self.batch_norm1(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x, edge_index))
        x = self.batch_norm2(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x, edge_index))
        
        # Global pooling
        gnn_mean = global_mean_pool(x, batch)
        gnn_max = global_max_pool(x, batch)
        gnn_features = torch.cat([gnn_mean, gnn_max], dim=1)
        
        # Descriptor branch
        desc_features = self.descriptor_branch(descriptors)
        
        # Fusion
        combined = torch.cat([gnn_features, desc_features], dim=1)
        combined = self.fusion(combined)
        
        return self.predictor(combined)


# ============================================================================
# PART 5: TRAINER
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
    
    def train(self, train_loader, val_loader, epochs=100):
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
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val R²: {val_metrics['r2']:.4f}")
        
        self.model.load_state_dict(torch.load('best_model.pt'))


# ============================================================================
# PART 6: MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("HYBRID GNN + DESCRIPTOR MODEL - Q1 PUBLICATION PIPELINE")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    print("\n[1/5] Loading ESOL dataset...")
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv'
    data = pd.read_csv(url)
    smiles = data['smiles'].values
    y = data['measured log solubility in mols per litre'].values
    
    # Normalize
    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / y_std
    
    # Split
    train_idx, test_idx = train_test_split(np.arange(len(smiles)), test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.125, random_state=42)  # 0.125 * 0.8 = 0.1
    
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Create datasets
    print("\n[2/5] Creating training dataset...")
    train_dataset = SimpleDataset(smiles[train_idx], y_norm[train_idx])
    
    print("\n[3/5] Creating validation dataset...")
    val_dataset = SimpleDataset(smiles[val_idx], y_norm[val_idx])
    
    print("\n[4/5] Creating test dataset...")
    test_dataset = SimpleDataset(smiles[test_idx], y_norm[test_idx])
    
    # Model
    print("\n[5/5] Training model...")
    model = HybridGNNModel(
        node_features_dim=20,
        hidden_dim=128,
        descriptor_dim=50,
        dropout=0.2,
        gnn_type='GCN'
    )
    
    trainer = ModelTrainer(model, device, learning_rate=0.001)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    trainer.train(train_loader, val_loader, epochs=100)
    
    # Final evaluation
    test_metrics = trainer.evaluate(test_loader)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test MAE:  {test_metrics['mae']:.4f}")
    print(f"Test R²:   {test_metrics['r2']:.4f}")
    print("=" * 60)
    
    return test_metrics


if __name__ == "__main__":
    results = main()