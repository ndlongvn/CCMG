"""Seeded response, drug, cell, and scaffold splits for regression.

All returned indices are response-row positions, independent of CSV index labels.
Group-based protocols keep each held-out group in exactly one partition.
"""

import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import shuffle

from data_process import DATA_DIR


def generate_scaffold(smiles, include_chirality=False):
    """Compute the Bemis–Murcko scaffold for a SMILES string."""
    return MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality
    )


class DataSplitter:
    def __init__(
        self,
        model_type,
        scenarios,
        cdr_reg_path=DATA_DIR / "CDR_n156813.txt",
        random_state=2022,
    ):
        if model_type != "regression":
            raise ValueError("Only regression is supported.")
        self.model_type = model_type
        self.scenarios = scenarios
        self.rs = random_state
        response = pd.read_csv(cdr_reg_path, sep="\t", index_col=0)
        # Dataset lookup uses iloc: labels in an input file need not be contiguous.
        self.CDR = shuffle(response.reset_index(drop=True), random_state=self.rs)

    def _fold(self, items, fold_idx, num_fold):
        if not 1 <= fold_idx <= num_fold or num_fold < 3:
            raise ValueError("Require num_fold >= 3 and 1 <= fold_idx <= num_fold.")
        kfold = KFold(n_splits=num_fold, shuffle=True, random_state=self.rs)
        for fold, (train_val, test) in enumerate(kfold.split(items), 1):
            if fold == fold_idx:
                train, valid = train_test_split(
                    train_val, test_size=1 / (num_fold - 1), random_state=self.rs
                )
                return train, valid, test
        raise ValueError(f"Could not construct fold {fold_idx}.")

    @staticmethod
    def _result(frames, return_train_y):
        train, valid, test = frames
        indices = tuple(frame.index.tolist() for frame in frames)
        if return_train_y:
            return ((train["lnIC50"].values, test["smiles"].values), *indices)
        return indices

    def split_discovery(self, fold_idx, num_fold=10, return_train_y=False):
        indices = self._fold(self.CDR, fold_idx, num_fold)
        frames = tuple(self.CDR.iloc[index] for index in indices)
        return self._result(frames, return_train_y)

    def _split_groups(self, column, fold_idx, num_fold, return_train_y):
        groups = self.CDR[column].unique()
        indices = self._fold(groups, fold_idx, num_fold)
        frames = tuple(
            self.CDR[self.CDR[column].isin(groups[index])] for index in indices
        )
        return self._result(frames, return_train_y)

    def split_blind_drug(self, fold_idx, num_fold=10, return_train_y=False):
        return self._split_groups("drug_id", fold_idx, num_fold, return_train_y)

    def split_blind_cell(self, fold_idx, num_fold=10, return_train_y=False):
        return self._split_groups("COSMIC_ID", fold_idx, num_fold, return_train_y)

    def split_blind_scaffold(self, fold_idx, num_fold=10, return_train_y=False):
        scaffold_smiles = {}
        for smiles in self.CDR["smiles"].unique():
            scaffold = generate_scaffold(smiles, include_chirality=True)
            scaffold_smiles.setdefault(scaffold, []).append(smiles)
        keys = list(scaffold_smiles)
        indices = self._fold(keys, fold_idx, num_fold)
        frames = []
        for index in indices:
            smiles = {smi for i in index for smi in scaffold_smiles[keys[i]]}
            frames.append(self.CDR[self.CDR["smiles"].isin(smiles)])
        return self._result(frames, return_train_y)

    def split_dataset(self, fold_idx, num_folds, return_train_y=False):
        methods = {
            "discovery": self.split_discovery,
            "blind_drug": self.split_blind_drug,
            "blind_cell": self.split_blind_cell,
            "blind_scaffold": self.split_blind_scaffold,
        }
        if self.scenarios not in methods:
            raise ValueError(f"Unknown splitting scenario: {self.scenarios}")
        return methods[self.scenarios](fold_idx, num_folds, return_train_y)
