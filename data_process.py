import numpy as np
import pandas as pd
import torch
from torch.utils import data
import os
import pickle
from itertools import chain, repeat

import networkx as nx
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import data
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
import json
from tqdm import tqdm
from rdkit import RDLogger
from rdkit.Chem import BRICS
RDLogger.DisableLog('rdApp.warning')  # Disable RDKit warnings
from multiprocessing import Pool, cpu_count
from functools import partial

from utils import onehot_encoding_unk
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import math
from rdkit.Chem import Descriptors, QED
from rdkit.Contrib.SA_Score.sascorer import calculateScore as sa_score
from multiprocessing import Pool, cpu_count
from rdkit import Chem,DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import os, multiprocessing, torch
import torch.nn as nn
from chemutils import motif_decomp, motif_decomp_fixed, wash_smiles
from gnn_model import GNN_Encoder
from torch_geometric.data import Batch
from torch.utils.data import DataLoader, Dataset


# Common list of atom types and special nodes
atom_types = ['B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'Pt', 'S', 'sub_structure', 'cell', 'other']

# Common list of hybridization types
hybridization_types = [
    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2, 'other'
]

    
def atom_attr(mol, explicit_H=True, use_chirality=True):
 
    feat = []
    for i, atom in enumerate(mol.GetAtoms()): # # + onehot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
        results = onehot_encoding_unk(
            atom.GetSymbol(),
            atom_types)  + [atom.GetDegree()] + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              onehot_encoding_unk(atom.GetHybridization(), hybridization_types) + [atom.GetIsAromatic()]
        results = results + onehot_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        results += [atom.HasProp('_ChiralityPossible')]

        results = np.array(results).astype(int)
        feat.append(results)
    
    feat = np.array(feat)
    
    return feat


def create_node_features(node_type, count):

    # Validate node_type
    valid_node_types = ['sub_structure', 'cell']
    if node_type not in valid_node_types:
        raise ValueError(f"Invalid node_type: {node_type}. Must be one of {valid_node_types}")
    
    
    # Generate feature vector
    results = (
        onehot_encoding_unk(node_type, atom_types) +  # One-hot encoding for node type
        [count] +  # Number of substructures or atoms
        [0, 0] +  # Placeholder features
        onehot_encoding_unk('other', hybridization_types) +  # One-hot encoding for hybridization
        [0] +  # Additional feature
        [1, 0, 0, 0, 0] +  # Fixed features
        [0]  # Final feature
    )
    
    # Convert to numpy array with integer type
    results = np.array([results]).astype(int)
    
    return results

def bond_attr_atom_subgraph_smiles(bond_type):

    bt = bond_type

    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bt == -999,  # -999 is used for sub-structure to atoms inside it
        bt == -1000,  # -1000 is used for atom to cell,
        bt == -1001, # -1001 is used for sub-structure to cell,
        0,
        0
    ]
    bond_feats = bond_feats + [0, 0, 0, 0, 1]

    return bond_feats



def bond_attr(mol, use_chirality=True):
    feat = []
    index = []
    n = mol.GetNumAtoms()
    for i in range(n):
        for j in range(n):
            if i != j:
                bond = mol.GetBondBetweenAtoms(i, j)
                if mol.GetAtomWithIdx(i).GetSymbol() == 'H' or mol.GetAtomWithIdx(j).GetSymbol() == 'H':
                    continue
                if bond is not None:
                    bt = bond.GetBondType()
                    bond_feats = [
                        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
                        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
                        bt == -999,  # -999 is used for sub-structure to atoms inside it
                        bt == -1000,  # -1000 is used for atom to cell,
                        bt == -1001, # -1001 is used for sub-structure to cell,
                        bond.GetIsConjugated(),
                        bond.IsInRing()
                    ]
                    if use_chirality:
                        bond_feats = bond_feats + onehot_encoding_unk(
                            str(bond.GetStereo()),
                            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", 'other'])
                    feat.append(bond_feats)
                    index.append([i, j])

    return np.array(index), np.array(feat)

def mol_to_graph(mol, brics, break_rings):
    
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {mol}")
    
    if brics:
        cliques = motif_decomp_fixed(mol, brics_only=True) # [[1,2], [3, 4]]
    elif break_rings:
        cliques = motif_decomp_fixed(mol, break_rings=True)
    else:
        cliques = motif_decomp_fixed(mol, break_rings=False)
    
    Chem.Kekulize(mol)

    data = mol_to_graph_data_obj(mol, cliques)
    return data
    
    
def mol_to_graph_data_obj(mol, cliques):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms


    atom_features_list = atom_attr(mol, explicit_H=True, use_chirality=True) # num nodes x num_atom_features
    x_nosuper = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_atom_features = 30
    num_bond_features = 14
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list, edge_features_list = bond_attr(mol, use_chirality=True)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index_nosuper = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr_nosuper = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index_nosuper = torch.empty((num_atom_features, 0), dtype=torch.long)
        edge_attr_nosuper = torch.empty((0, num_bond_features), dtype=torch.long)

    # print(f"x_nosuper shape: {x_nosuper.shape}, edge_index_nosuper shape: {edge_index_nosuper.shape}, edge_attr_nosuper shape: {edge_attr_nosuper.shape}")

    num_atoms = x_nosuper.size(0)
    # print(f"num_atoms: {num_atoms}, num_bonds: {edge_index_nosuper.size(1)}")
    #add motif 

    for i, clique in enumerate(cliques):
        if not all(0 <= idx < num_atoms for idx in clique):
            raise ValueError(f"Clique {i} contains invalid indices: {clique} (must be in [0, {num_atoms}))")
    num_motif = len(cliques) 

    if num_motif > 0: # can break to many sub-structures
        # create super node
        super_x = create_node_features("cell", num_motif)  # (num_motif + 1) x num_atom_features
        super_x = torch.tensor(super_x, dtype=torch.long).to(x_nosuper.device)

        motif_x = []
        # create motif nodes
        for sub_graph in cliques:
            motif = create_node_features("sub_structure", len(sub_graph))  # (num_motif + 2) x num_atom_features
            # print(motif.shape)
            motif_x.append(motif)
        motif_x = torch.tensor(np.array(motif_x), dtype=torch.long).to(x_nosuper.device).squeeze(1)  # num_motif x num_atom_features

        # create edge index and edge attr
        motif_edge_index = []
        for k, motif in enumerate(cliques):
            motif_edge_index = motif_edge_index + [[i, num_atoms+k] for i in motif]
        motif_edge_index = torch.tensor(np.array(motif_edge_index).T, dtype=torch.long).to(edge_index_nosuper.device)

        super_edge_index = [[num_atoms+i, num_atoms+num_motif] for i in range(num_motif)]
        super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(edge_index_nosuper.device)

        # cat edge index
        edge_index = torch.cat((edge_index_nosuper, motif_edge_index, super_edge_index), dim=1)

        motif_edge_attr = []
        for i in range(len(motif_edge_index.T)):
            motif_edge_attr.append(bond_attr_atom_subgraph_smiles(-999))
        motif_edge_attr = torch.tensor(np.array(motif_edge_attr), dtype=torch.long).to(edge_attr_nosuper.device)
        
        # add super edge attr
        super_edge_attr = []
        for i in range(num_motif):
            super_edge_attr.append(bond_attr_atom_subgraph_smiles(-1001))
        super_edge_attr = torch.tensor(np.array(super_edge_attr), dtype=torch.long).to(edge_attr_nosuper.device)

        # cat edge attr
        edge_attr = torch.cat((edge_attr_nosuper, motif_edge_attr, super_edge_attr), dim=0)

        x = torch.cat((x_nosuper, motif_x, super_x), dim=0)

    else: # no sub-structure at all
        super_x = create_node_features("cell", num_atoms)  # num_atoms x num_atom_features
        super_x = torch.tensor(super_x, dtype=torch.long).to(x_nosuper.device)

        super_edge_index = [[i, num_atoms] for i in range(num_atoms)]
        super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(edge_index_nosuper.device)
        # create edge index and edge attr

        # cat edge index
        edge_index = torch.cat((edge_index_nosuper, super_edge_index), dim=1)

        # add super edge attr
        super_edge_attr = []
        for i in range(num_atoms):
            super_edge_attr.append(bond_attr_atom_subgraph_smiles(-1001))
        super_edge_attr = torch.tensor(np.array(super_edge_attr), dtype=torch.long).to(edge_attr_nosuper.device)

        # cat edge attr
        edge_attr = torch.cat((edge_attr_nosuper, super_edge_attr), dim=0)

        # cat x
        x = torch.cat((x_nosuper, super_x), dim=0)


    num_part = (num_atoms, num_motif, 2)
    data = Data(x=x.to(torch.float), edge_index=edge_index, edge_attr=edge_attr.to(torch.float), num_part = num_part)
    return data




def read_smiles(data_path):
    smiles_data = []
    with open(data_path, 'r') as f:
        for line in tqdm(f, desc="read"):
            smiles_data.append(line.strip())
    return smiles_data


class MoleculeDataset(Dataset):
    def __init__(self,
                 root,
                 original_dataset_path,
                 indices,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 scaler=None,
                 dataset='all',
                 empty=False,
                 config=None):

        # ---------- define everything *before* super().__init__ ----------
        self.config  = config
        self.dataset = dataset
        self.root    = root
        self.scaler = scaler
        self.indices = indices

        if original_dataset_path is not None:

            self.original_dataset = pd.read_csv(original_dataset_path, sep='\t', index_col=0)
            self.original_dataset.index = self.original_dataset.index.astype(str)  # Ensure index is string type for consistency
            self.drug_df = self.original_dataset.reset_index()
            if 'regression' in config['model_type']:
                self.labels = self.drug_df['lnIC50'].values
            else:
                raise ValueError("Invalid model_type. Choose 'regression'.")
            self.list_IDs = self.drug_df.index.values
            
            if config['wash_smiles']:
                smiles = self.drug_df['smiles'].unique()
                smiles_dict = {smi: wash_smiles(smi) for smi in tqdm(smiles, desc="Washing SMILES")}
                self.drug_df['smiles'] = self.drug_df['smiles'].map(smiles_dict)
                # self.drug_df['smiles'] = self.drug_df['smiles'].apply(wash_smiles)
            self.drug_df['smiles'] = self.drug_df['smiles'].astype(str)
            
    
            if self.config['external_dataset'] == 'None':
                self.rna_data = pd.read_csv('./data/GDSC/data_processed/RNA_n18451_1018_zscore.csv',index_col=0)
                self.rna_data.columns = [name.split('.')[0][1:] for name in self.rna_data.columns.values]
                self.rna_data = self.rna_data.T
                self.genetic = pd.read_csv('./data/GDSC/data_processed/Genetic_features_n969_735.txt',sep='\t',index_col=0)
                self.mrna = pd.read_csv('./data/GDSC/data_processed/mrna_n20617_1028_zscore.csv',index_col=0)
                self.mrna = self.mrna.T
    
    
            required_cols = ['smiles', 'lnIC50' if 'regression' in config['model_type']  else 'Label']
            if self.config['screening'] == 'TCGA':
                required_cols.append('patient.arr')
            elif self.config['external_dataset'] == 'None':
                required_cols.extend(['assay_name', 'COSMIC_ID', 'cell_type'])
            elif self.config['external_dataset'] == 'TCGA':
                required_cols.append('patient.arr')
            elif self.config['external_dataset'] == 'CCLE':
                required_cols.extend(['DepMap_ID', 'cell_lines'])
    
            missing_cols = [col for col in required_cols if col not in self.drug_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in drug_df: {missing_cols}")
        
        super(MoleculeDataset, self).__init__()
        
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        self.process( brics=config['brics'], 
                     break_rings=config['break_rings'])
            
    def set_scaler(self, scaler):
        self.scaler = scaler
    
    def __len__(self):
        # return len(self.data_list)
        return len(self.indices)

    def __getitem__(self, idx):
        real_index = self.indices[idx]
        real_index = self.list_IDs[real_index]
        # graph data
        smiles = self.drug_df.iloc[real_index]['smiles']
        data = self.smiles_dict[smiles].clone()
        data.id = torch.tensor([real_index])
        y = self.drug_df.iloc[real_index]['lnIC50'] if self.config['model_type'] == 'regression' else self.drug_df.iloc[real_index]['Label']
        if len(np.array(y).shape) == 2:
            data.y = torch.tensor(y.reshape(1, -1), dtype=torch.float)
        else:
            data.y = torch.tensor([y], dtype=torch.float)
            
        # omics data
        v_rna = np.array(self.rna_data.loc[self.drug_df.iloc[real_index]['assay_name']]).reshape(1, -1)
        v_genetic = np.array(self.genetic.loc[int(self.drug_df.iloc[real_index]['COSMIC_ID'])]).reshape(1, -1)
        v_mrna = np.array(self.mrna.loc[self.drug_df.iloc[real_index]['cell_type']]).reshape(1, -1)
        # data.fingerprint = torch.tensor(self.smiles2morgan(self.drug_df.iloc[real_index]['smiles']), dtype=torch.float)

        # Convert to tensors
        data.v_rna = torch.tensor(v_rna, dtype=torch.float)
        data.v_genetic = torch.tensor(v_genetic, dtype=torch.float)
        data.v_mrna = torch.tensor(v_mrna, dtype=torch.float)
        if self.scaler:
            data.y = torch.tensor(self.scaler.transform(data.y[None].numpy())[0], dtype=torch.float)
        return data

        
    def process(self, brics=False, break_rings=False):
        data_list = []

        smiles_dict = {}
        num_unique_smiles = self.drug_df['smiles'].unique()
        print(f"Number of unique SMILES: {len(num_unique_smiles)}")
        n = 0
        for smi in tqdm(num_unique_smiles, desc="Processing molecules"): 
            rdkit_mol = Chem.MolFromSmiles(smi)
            if rdkit_mol is None: 
                print(f"Invalid SMILES: {smi}")
                continue
            data = mol_to_graph(rdkit_mol, brics=brics, break_rings=break_rings)
            smiles_dict[smi] = data
            if data.num_part[1]>0:
                n+=1
        print("Percentage of mols have motif: ", n/len(num_unique_smiles))
        
        self.smiles_dict = smiles_dict


def create_all_datasets():
    #### create dataset

    model_type = ['regression']
    screening = ['None']
    external_dataset = ['None']
    for model in model_type:
        for screen in screening:
            for external in external_dataset:
                config = {
                    'model_type': model,  # regression or classification
                    'screening': screen,  # 'None': traning TransCDR using GDSC dataset; 'TCGA': screening TCGA dataset  # the pre-trained graph model
                    'external_dataset': external  # 'None':traning TransCDR using GDSC dataset; 'CCLE', 'TCGA': test on external_dataset,
                    
                }
                if 'regression' in model:
                    original_dataset_path = './data/GDSC/data_processed/CDR_n156813.txt'
                else:
                    raise ValueError("Invalid model type")
                root = os.path.join("data", model)
                os.makedirs(root, exist_ok=True)
                os.makedirs(root + "/processed", exist_ok=True)
                dataset = MoleculeDataset(root, config=config, 
                                        original_dataset_path=original_dataset_path,
                                        dataset=model)
                print(dataset)




# test MoleculeDataset object
if __name__ == "__main__":
    create_all_datasets()
            