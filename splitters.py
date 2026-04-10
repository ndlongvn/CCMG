import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, train_test_split
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from chemutils import motif_decomp, get_mol
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------------- #
#                               MAIN CLASS                                   #
# -------------------------------------------------------------------------- #

def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain assert from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


class DataSplitter:
    """
    Wrapper tách dữ liệu CV-10 theo nhiều kịch bản (warm-start, cold-drug, ...).
    - Chỉ ghi INDEX ra train.txt / val.txt / test.txt.
    - Các đường dẫn dữ liệu gốc có thể truyền hoặc sửa trực tiếp.
    """

    # ---- khởi tạo ---------------------------------------------------------
    def __init__(self,
                 model_type: str,
                 scenarios: str,
                 n_folds: int = 10,
                 n_clusters: int = 10,
                 n_sampling: int = 1,
                 cdr_reg_path: str = "./data/GDSC/data_processed/CDR_n156813_filter.txt",
                 cdr_cls_path: str = "./data/GDSC/data_processed/CDR_bi_n154603.txt",
                 random_state: int = 2022):
        self.model_type    = model_type
        self.scenarios     = scenarios
        self.n_clusters    = n_clusters
        self.n_sampling    = n_sampling
        self.rs            = random_state
        self._load_cdr(cdr_reg_path, cdr_cls_path)


    # ------------------------------------------------------------------ #
    #                         LOAD & PREPROCESS                          #
    # ------------------------------------------------------------------ #
    def _load_cdr(self, reg_path, cls_path):
        """Đọc CDR và (nếu classification) down-sample âm/dương."""
        if self.model_type == "regression":
            self.CDR = shuffle(pd.read_csv(reg_path, sep="\t", index_col=0),
                               random_state=self.rs)
        else:
            cdr = shuffle(pd.read_csv(cls_path, sep="\t", index_col=0),
                          random_state=self.rs)
            pos = cdr[cdr.Label == 1]
            neg = cdr[cdr.Label == 0].sample(
                n=self.n_sampling * len(pos), random_state=self.rs)
            self.CDR = shuffle(pd.concat([pos, neg]), random_state=self.rs)

    # ------------------------------------------------------------------ #
    #                         SPLIT FUNCTIONS                            #
    # ------------------------------------------------------------------ #

    def split_final(self, fold_idx, num_fold=10, return_train_y=False):
        # self.CDR['motif'] = self.CDR['smiles'].apply(lambda g: len(motif_decomp(get_mol(g))))
        # labels = self.CDR['motif'].values

        # Step 1: extract unique SMILES
        unique_smiles = self.CDR['smiles'].unique()
        
        # Step 2: compute motif length only once per unique SMILES
        motif_map = {
            s: len(motif_decomp(get_mol(s))) for s in unique_smiles
        }
        
        # Step 3: map back to DataFrame
        self.CDR['motif'] = self.CDR['smiles'].map(motif_map)
        
        # Labels for stratified split etc.
        labels = self.CDR['motif'].values


        # 1) 90/10 split
        X_train_val, X_test = train_test_split(
            self.CDR, test_size=0.10, stratify=labels, random_state=42
        )
        
        # 2) 80/10 split from the remainder → use labels from the subset
        labels_train_val = X_train_val['motif'].values
        X_train, X_val = train_test_split(
            X_train_val, test_size=1/9, stratify=labels_train_val, random_state=42
        )

        # print("🔹 Train motif counts:")
        # print(X_train['motif'].value_counts(normalize=True))
        
        # print("\n🔹 Validation motif counts:")
        # print(X_val['motif'].value_counts(normalize=True))
        
        # print("\n🔹 Test motif counts:")
        # print(X_test['motif'].value_counts(normalize=True))
        

        if return_train_y:
            if self.model_type == 'regression':
                return (X_train['lnIC50'].values, X_test['smiles'].values), list(X_train.index), list(X_val.index), list(X_test.index)
            else:
                return (X_train['lnIC50'].values, X_test['smiles'].values), list(X_train.index), list(X_val.index), list(X_test.index)

        return list(X_train.index), list(X_val.index), list(X_test.index)
        
                
    def split_discovery(self, fold_idx, num_fold=10, return_train_y=False):
        kf = KFold(n_splits=num_fold, shuffle=True, random_state=self.rs)

        for fold, (tr, te) in enumerate(kf.split(self.CDR), 1):
            if fold_idx == fold:
                tr_val, te_df = self.CDR.iloc[tr], self.CDR.iloc[te]
                # Use 1/(num_fold - 1) to ensure train:val:test ~ (num_fold-2):(1):(1)
                val_ratio = 1 / (num_fold - 1)
                tr_df, val_df = train_test_split(tr_val, test_size=val_ratio, random_state=self.rs)

                if return_train_y:
                    if self.model_type == 'regression':
                        return (tr_df['lnIC50'].values, te_df['smiles'].values), list(tr_df.index), list(val_df.index), list(te_df.index)
                    else:
                        return (tr_df['lnIC50'].values, te_df['smiles'].values), list(tr_df.index), list(val_df.index), list(te_df.index)

                return list(tr_df.index), list(val_df.index), list(te_df.index)
            

    # ---------- cold-drug --------------------------------------------------

    def split_blind_drug(self, fold_idx, num_fold=10, return_train_y=False):
        drugs = pd.DataFrame({"drug_id": self.CDR.drug_id.unique()})
        kf = KFold(n_splits=num_fold, shuffle=True, random_state=self.rs)

        for fold, (tr, te) in enumerate(kf.split(drugs), 1):
            if fold_idx == fold:
                tr_val, te_drug   = drugs.iloc[tr], drugs.iloc[te]
                tr_drug, val_drug = train_test_split(tr_val, test_size=1/(num_fold-1),
                                                    random_state=self.rs)

                tr_df  = self.CDR.loc[self.CDR.drug_id.isin(tr_drug.drug_id)]
                val_df = self.CDR.loc[self.CDR.drug_id.isin(val_drug.drug_id)]
                te_df  = self.CDR.loc[self.CDR.drug_id.isin(te_drug.drug_id)]
                
                if return_train_y:
                    if self.model_type == 'regression':
                        return (tr_df['lnIC50'].values, te_df['smiles'].values), list(tr_df.index), list(val_df.index), list(te_df.index)
                    else:
                        return (tr_df['lnIC50'].values, te_df['smiles'].values), list(tr_df.index), list(val_df.index), list(te_df.index)

                return list(tr_df.index), list(val_df.index), list(te_df.index)

    # ---------- cold-cell --------------------------------------------------
    def split_blind_cell(self, fold_idx, num_fold=10, return_train_y=False):
        cells = pd.DataFrame({"COSMIC_ID": self.CDR.COSMIC_ID.unique()})
        kf = KFold(n_splits=num_fold, shuffle=True, random_state=self.rs)

        for fold, (tr, te) in enumerate(kf.split(cells), 1):
            if fold_idx == fold:

                tr_val, te_cell   = cells.iloc[tr], cells.iloc[te]
                tr_cell, val_cell = train_test_split(tr_val, test_size=1/(num_fold-1),
                                                    random_state=self.rs)

                tr_df  = self.CDR.loc[self.CDR.COSMIC_ID.isin(tr_cell.COSMIC_ID)]
                val_df = self.CDR.loc[self.CDR.COSMIC_ID.isin(val_cell.COSMIC_ID)]
                te_df  = self.CDR.loc[self.CDR.COSMIC_ID.isin(te_cell.COSMIC_ID)]
                
                if return_train_y:
                    if self.model_type == 'regression':
                        return (tr_df['lnIC50'].values, te_df['smiles'].values), list(tr_df.index), list(val_df.index), list(te_df.index)
                    else:
                        return (tr_df['lnIC50'].values, te_df['smiles'].values), list(tr_df.index), list(val_df.index), list(te_df.index)

                return list(tr_df.index), list(val_df.index), list(te_df.index)
            
    # ---------- cold-scaffold --------------------------------------------------
            
    def split_blind_scaffold(self, fold_idx, num_fold=10, return_train_y=False):
        # Get unique SMILES strings
        smiles_list = self.CDR['smiles'].unique()
        
        # Group indices by scaffold
        all_scaffolds = {}
        for i, smiles in enumerate(smiles_list):
            scaffold = generate_scaffold(smiles, include_chirality=True)
            all_scaffolds.setdefault(scaffold, []).append(smiles)
        
        # Convert to sorted lists for consistency
        all_scaffolds = {k: sorted(v) for k, v in all_scaffolds.items()}
        scaffold_keys = list(all_scaffolds.keys())
        
        # Perform K-fold split
        kf = KFold(n_splits=num_fold, random_state=self.rs, shuffle=True)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(scaffold_keys), 1):
            if fold == fold_idx:
                # Split training indices into train and validation
                train_idx, val_idx = train_test_split(
                    train_idx, 
                    test_size=1/(num_fold-1), 
                    random_state=self.rs
                )
                
                # Map indices to SMILES using sets for efficiency
                train_smiles = set()
                val_smiles = set()
                test_smiles = set()
                
                for idx in train_idx:
                    train_smiles.update(all_scaffolds[scaffold_keys[idx]])
                for idx in val_idx:
                    val_smiles.update(all_scaffolds[scaffold_keys[idx]])
                for idx in test_idx:
                    test_smiles.update(all_scaffolds[scaffold_keys[idx]])
                    
                # print(train_smiles)
                
                # Create dataframes using vectorized operations
                train_df = self.CDR[self.CDR['smiles'].isin(train_smiles)]
                val_df = self.CDR[self.CDR['smiles'].isin(val_smiles)]
                test_df = self.CDR[self.CDR['smiles'].isin(test_smiles)]
                
                if return_train_y:
                    target_col = 'lnIC50' if self.model_type == 'regression' else 'Label'
                    return (train_df[target_col].values , test_df['smiles'].values), train_df.index.tolist(), val_df.index.tolist(), test_df.index.tolist()
                
                return train_df.index.tolist(), val_df.index.tolist(), test_df.index.tolist()
        
            
    def split_dataset(self, fold_idx, num_folds, return_train_y=False):
        if self.scenarios == 'discovery':
            return self.split_discovery(fold_idx, num_folds, return_train_y)
        elif self.scenarios == 'final':
            return self.split_final(fold_idx, num_folds, return_train_y)
        elif self.scenarios == 'blind_drug':
            return self.split_blind_drug(fold_idx, num_folds, return_train_y)
        elif self.scenarios == 'blind_cell':
            return self.split_blind_cell(fold_idx, num_folds, return_train_y)
        elif self.scenarios == 'blind_scaffold':
            return self.split_blind_scaffold(fold_idx, num_folds, return_train_y)
        else:
            raise ValueError("Invalid splitting scenarios.")
            


if __name__ == "__main__":
    splitter = DataSplitter(
        model_type="classification",
        scenarios="warm start",
    )
    x, y, z = splitter.split_discovery(1)
    print(f"Train: {len(x)}, Val: {len(y)}, Test: {len(z)}")
    print(x[:5])
    print(y[:5])
    print(z[:5])
