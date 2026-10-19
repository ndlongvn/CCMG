"""Molecular graph features and GDSC drug-response datasets.

Graphs retain the original directed atom-to-motif-to-cell connections. The cell
node is always last; the model relies on this ordering when inserting omics.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from chemutils import motif_decomp_fixed, wash_smiles
from utils import onehot_encoding_unk

DATA_DIR = Path(__file__).resolve().parent / "data" / "GDSC" / "data_processed"

# Common list of atom types and special nodes
atom_types = [
    "B",
    "Br",
    "C",
    "Cl",
    "F",
    "I",
    "N",
    "O",
    "P",
    "Pt",
    "S",
    "sub_structure",
    "cell",
    "other",
]

# Common list of hybridization types
hybridization_types = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    "other",
]


def atom_attr(mol):
    """Return the original 30 atom features, including virtual-node flags."""
    feat = []
    for atom in mol.GetAtoms():
        results = (
            onehot_encoding_unk(atom.GetSymbol(), atom_types)
            + [atom.GetDegree()]
            + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
            + onehot_encoding_unk(atom.GetHybridization(), hybridization_types)
            + [atom.GetIsAromatic()]
        )
        results = results + onehot_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        results += [atom.HasProp("_ChiralityPossible")]

        results = np.array(results).astype(int)
        feat.append(results)

    feat = np.array(feat)

    return feat


def create_node_features(node_type, count):
    """Create a virtual node with the same feature width as a physical atom."""
    # Validate node_type
    valid_node_types = ["sub_structure", "cell"]
    if node_type not in valid_node_types:
        raise ValueError(
            f"Invalid node_type: {node_type}. Must be one of {valid_node_types}"
        )

    # Generate feature vector
    results = (
        onehot_encoding_unk(node_type, atom_types)  # One-hot encoding for node type
        + [count]  # Number of substructures or atoms
        + [0, 0]  # Placeholder features
        + onehot_encoding_unk(
            "other", hybridization_types
        )  # One-hot encoding for hybridization
        + [0]  # Additional feature
        + [1, 0, 0, 0, 0]  # Fixed features
        + [0]  # Final feature
    )

    # Convert to numpy array with integer type
    results = np.array([results]).astype(int)

    return results


def bond_attr_atom_subgraph_smiles(bond_type):
    """Encode a virtual-edge type using the original 14-feature layout."""
    bt = bond_type

    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bt == -999,  # -999 is used for sub-structure to atoms inside it
        bt == -1000,  # -1000 is used for atom to cell,
        bt == -1001,  # -1001 is used for sub-structure to cell,
        0,
        0,
    ]
    bond_feats = bond_feats + [0, 0, 0, 0, 1]

    return bond_feats


def bond_attr(mol, use_chirality=True):
    """Return directed bonds in source/target order, matching the original graph."""
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (
            mol.GetAtomWithIdx(i).GetSymbol() == "H"
            or mol.GetAtomWithIdx(j).GetSymbol() == "H"
        ):
            continue
        edges.extend([(i, j, bond), (j, i, bond)])
    edges.sort(key=lambda edge: edge[:2])
    indices, features = [], []
    for i, j, bond in edges:
        bt = bond.GetBondType()
        values = [
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            False,
            False,
            False,
            bond.GetIsConjugated(),
            bond.IsInRing(),
        ]
        if use_chirality:
            values += onehot_encoding_unk(
                str(bond.GetStereo()),
                ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "other"],
            )
        indices.append([i, j])
        features.append(values)
    return np.asarray(indices, dtype=int).reshape(-1, 2), np.asarray(
        features, dtype=int
    ).reshape(-1, 14 if use_chirality else 9)


def mol_to_graph(mol, brics):
    """Decompose, kekulize, and encode an RDKit molecule or SMILES string."""
    if isinstance(mol, str):
        smiles = mol
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Cannot construct a graph from an empty molecule.")

    if brics:
        cliques = motif_decomp_fixed(mol, brics_only=True)
    else:
        cliques = motif_decomp_fixed(mol, brics_only=False)

    Chem.Kekulize(mol)

    return mol_to_graph_data_obj(mol, cliques)


def mol_to_graph_data_obj(mol, cliques):
    """Build atoms, motifs, and a final cell node with directed virtual edges.

    Physical bonds are bidirectional. Atoms point to motifs and motifs point to
    the cell. Without motifs, atoms point directly to the cell using the original
    -1001 edge features. ``num_part`` retains the legacy metadata tuple.
    """
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Cannot construct a graph from an empty molecule.")
    num_atoms, num_motifs = mol.GetNumAtoms(), len(cliques)
    for i, clique in enumerate(cliques):
        if not clique or not all(0 <= index < num_atoms for index in clique):
            raise ValueError(f"Invalid atom indices in clique {i}: {clique}")

    atom_features = torch.tensor(atom_attr(mol), dtype=torch.float)
    bond_indices, bond_features = bond_attr(mol)
    edge_index = torch.tensor(bond_indices.T, dtype=torch.long)
    edge_attr = torch.tensor(bond_features, dtype=torch.float)
    virtual_nodes, virtual_edges, virtual_features = [], [], []
    if num_motifs:
        for motif_index, clique in enumerate(cliques):
            virtual_nodes.append(create_node_features("sub_structure", len(clique))[0])
            for atom_index in clique:
                virtual_edges.append([atom_index, num_atoms + motif_index])
                virtual_features.append(bond_attr_atom_subgraph_smiles(-999))
        cell_index = num_atoms + num_motifs
        for motif_index in range(num_motifs):
            virtual_edges.append([num_atoms + motif_index, cell_index])
            virtual_features.append(bond_attr_atom_subgraph_smiles(-1001))
    else:
        for atom_index in range(num_atoms):
            virtual_edges.append([atom_index, num_atoms])
            virtual_features.append(bond_attr_atom_subgraph_smiles(-1001))
    virtual_nodes.append(create_node_features("cell", num_motifs or num_atoms)[0])
    x = torch.cat(
        [atom_features, torch.tensor(np.asarray(virtual_nodes), dtype=torch.float)]
    )
    edge_index = torch.cat(
        [edge_index, torch.tensor(virtual_edges, dtype=torch.long).T], dim=1
    )
    edge_attr = torch.cat(
        [edge_attr, torch.tensor(virtual_features, dtype=torch.float)], dim=0
    )
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_part=(num_atoms, num_motifs, 2),
    )


class MoleculeDataset(Dataset):
    """Load response/omics tables once and cache graphs by normalized SMILES.

    ``indices`` are zero-based response-row positions produced by DataSplitter.
    Invalid molecules fail explicitly instead of silently changing the split.
    """

    def __init__(self, root, original_dataset_path, indices, scaler=None, config=None):
        super().__init__()
        self.config = config or {}
        self.root = root
        self.scaler = scaler
        self.indices = list(indices)
        self.drug_df = pd.read_csv(
            original_dataset_path, sep="\t", index_col=0
        ).reset_index(drop=True)
        required_cols = ["smiles", "lnIC50", "assay_name", "COSMIC_ID", "cell_type"]
        missing_cols = [col for col in required_cols if col not in self.drug_df]
        if missing_cols:
            raise ValueError(f"Missing required response columns: {missing_cols}")
        if self.config.get("model_type", "regression") != "regression":
            raise ValueError("Only regression is supported.")

        if self.config.get("wash_smiles", True):
            smiles = self.drug_df["smiles"].unique()
            normalized = {
                smi: wash_smiles(smi) for smi in tqdm(smiles, desc="Washing SMILES")
            }
            self.drug_df["smiles"] = self.drug_df["smiles"].map(normalized)
        invalid = self.drug_df["smiles"].isna() | self.drug_df["smiles"].eq("")
        if invalid.any():
            raise ValueError(
                f"Invalid SMILES at response row positions: {self.drug_df.index[invalid].tolist()[:10]}"
            )

        data_dir = Path(self.config.get("data_dir", DATA_DIR))
        self.rna_data = pd.read_csv(
            data_dir / "RNA_n18451_1018_zscore.csv", index_col=0
        )
        self.rna_data.columns = [
            name.split(".")[0][1:] for name in self.rna_data.columns
        ]
        self.rna_data = self.rna_data.T
        self.genetic = pd.read_csv(
            data_dir / "Genetic_features_n969_735.txt", sep="\t", index_col=0
        )
        self.mrna = pd.read_csv(data_dir / "mrna_n20617_1028_zscore.csv", index_col=0).T
        self.process(brics=self.config.get("brics", False))

    def set_scaler(self, scaler):
        self.scaler = scaler

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_index = self.indices[idx]
        row = self.drug_df.iloc[real_index]
        data = self.smiles_dict[row["smiles"]].clone()
        data.id = torch.tensor([real_index])
        target = float(row["lnIC50"])
        if self.scaler is not None:
            target = self.scaler.transform([[target]])[0, 0]
        data.y = torch.tensor([target], dtype=torch.float)
        data.v_rna = torch.tensor(
            np.asarray(self.rna_data.loc[row["assay_name"]]).reshape(1, -1),
            dtype=torch.float,
        )
        data.v_genetic = torch.tensor(
            np.asarray(self.genetic.loc[int(row["COSMIC_ID"])]).reshape(1, -1),
            dtype=torch.float,
        )
        data.v_mrna = torch.tensor(
            np.asarray(self.mrna.loc[row["cell_type"]]).reshape(1, -1),
            dtype=torch.float,
        )
        return data

    def process(self, brics=False):
        """Cache one graph per normalized SMILES without changing row order."""
        smiles_dict = {}
        num_unique_smiles = self.drug_df["smiles"].unique()
        print(f"Number of unique SMILES: {len(num_unique_smiles)}")
        n = 0
        for smi in tqdm(num_unique_smiles, desc="Processing molecules"):
            rdkit_mol = Chem.MolFromSmiles(smi)
            if rdkit_mol is None:
                raise ValueError(f"Cannot build graph for invalid SMILES: {smi!r}")
            data = mol_to_graph(rdkit_mol, brics=brics)
            smiles_dict[smi] = data
            if data.num_part[1] > 0:
                n += 1
        print(
            "Fraction of molecules with motifs:",
            n / len(num_unique_smiles) if len(num_unique_smiles) else 0.0,
        )

        self.smiles_dict = smiles_dict
