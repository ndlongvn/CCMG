import random
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
import torch_geometric.transforms as T
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import sparse as sp
import torch
import numpy as np
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import math
from typing import Any
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in PyTorch, numpy and
    Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def onehot_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [1 if x == s else 0 for s in allowable_set]


def onehot_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [1 if x == s else 0 for s in allowable_set]





# copy from xiong et al. attentivefp
class ScaffoldGenerator(object):
    """
    Generate molecular scaffolds.
    Parameters
    ----------
    include_chirality : : bool, optional (default False)
        Include chirality in scaffolds.
    """

    def __init__(self, include_chirality=False):
        self.include_chirality = include_chirality

    def get_scaffold(self, mol):
        """
        Get Murcko scaffolds for molecules.
        Murcko scaffolds are described in DOI: 10.1021/jm9602928. They are
        essentially that part of the molecule consisting of rings and the
        linker atoms between them.
        Parameters
        ----------
        mols : array_like
            Molecules.
        """
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=self.include_chirality)

# copy from xiong et al. attentivefp
def generate_scaffold(smiles, include_chirality=False):
    """Compute the Bemis-Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    engine = ScaffoldGenerator(include_chirality=include_chirality)
    scaffold = engine.get_scaffold(mol)
    return scaffold

# copy from xiong et al. attentivefp
def split(scaffolds_dict, smiles_tasks_df, sample_size, random_seed=0):
    # Chia làm sao đảm bảo tý lệ của cần chia ví dụ 0.9<10%<1.1 
    count = 0
    optimal_count = 0.1 * len(smiles_tasks_df)
    while (count < optimal_count * 0.8 or count > optimal_count * 1.2):
        random_seed += 1
        random.seed(random_seed)
        scaffold = random.sample(list(scaffolds_dict.keys()), sample_size)
        count = sum([len(scaffolds_dict[scaffold]) for scaffold in scaffold])
        index = [index for scaffold in scaffold for index in scaffolds_dict[scaffold]]
    return scaffold, index

# copy from xiong et al. attentivefp
# def scaffold_randomized_spliting(smiles_tasks_df, random_seed=8, ratio_test= 0.1, ration_valid= 0.1):
#     print('generating scaffold......')
#     scaffold_list = []
#     all_scaffolds_dict = {}
#     for index, smiles in enumerate(smiles_tasks_df):
#         scaffold = generate_scaffold(smiles)
#         scaffold_list.append(scaffold)
#         if scaffold not in all_scaffolds_dict:
#             all_scaffolds_dict[scaffold] = [index]
#         else:
#             all_scaffolds_dict[scaffold].append(index)

#     samples_size_test = int(len(all_scaffolds_dict.keys()) * ratio_test)
#     samples_size_val = int(len(all_scaffolds_dict.keys()) * ration_valid * (1-ratio_test))
#     test_scaffold, test_index = split(all_scaffolds_dict, smiles_tasks_df, samples_size_test,
#                                       random_seed=random_seed)
#     training_scaffolds_dict = {x: all_scaffolds_dict[x] for x in all_scaffolds_dict.keys() if x not in test_scaffold}
#     valid_scaffold, valid_index = split(training_scaffolds_dict, smiles_tasks_df, samples_size_val,
#                                         random_seed=random_seed)

#     training_scaffolds_dict = {x: training_scaffolds_dict[x] for x in training_scaffolds_dict.keys() if
#                                x not in valid_scaffold}
#     train_index = []
#     for ele in training_scaffolds_dict.values():
#         train_index += ele
#     assert len(train_index) + len(valid_index) + len(test_index) == len(smiles_tasks_df)

#     return train_index, valid_index, test_index


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold

from collections import defaultdict
def scaffold_randomized_spliting(dataset, random_seed= 8, ratio_test= 0.1, ration_valid= 0.1):
    print('generating scaffold......')

    rng = np.random.RandomState(random_seed)
    # if isinstance(dataset, str):
    #     dataset= pd.read_csv(dataset)
    # try:
    #     smiles_list= dataset['smiles'].values
    # except:
    #     smiles_list= dataset['SMILES'].values
    smiles_list = dataset
    scaffolds = defaultdict(list)
    for ind, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)
    idxs= list(scaffolds.keys())
    idxs = rng.permutation(idxs)
    scaffold_sets = [scaffolds[idx] for idx in idxs]

    n_total_valid = int(ration_valid * len(dataset) * (1-ratio_test))
    n_total_test = int(ratio_test * len(dataset))
    print('Num train: {}, Num val {}, Num test {}'.format(len(smiles_list)-n_total_test-n_total_valid, n_total_valid, n_total_test))
    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        elif len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0
    assert len(set(train_idx)) + len(set(test_idx))+ len(set(valid_idx)) == len(smiles_list), 'total not match'

    return train_idx, valid_idx, test_idx





class Classification_Module(nn.Module):
    def __init__(self, num_features_xd=256, output_dim=1, dropout = 0.1, regression=False, pool='max'):
        super(Classification_Module, self).__init__()
        self.fc_g1   = torch.nn.Linear(num_features_xd, num_features_xd)
        self.fc_g2   = torch.nn.Linear(num_features_xd, num_features_xd//2)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out     = nn.Linear(num_features_xd//2, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.regression = regression
            
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool
        

    def forward(self, x, batch):
        # Maxpool
        x   = self.pool(x, batch)
        #MLP
        x   = self.relu(self.fc_g1(x))
        x   = self.dropout(x)
        x   = self.fc_g2(x)
        x   = self.dropout(x)
        out = self.out(x).view(-1)
        if self.regression:
            return out
        out = self.sigmoid(out)
        return out



def uniform(size: int, value: Any):
    if isinstance(value, Tensor):
        bound = 1.0 / math.sqrt(size)
        value.data.uniform_(-bound, bound)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            uniform(size, v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            uniform(size, v)


def kaiming_uniform(value: Any, fan: int, a: float):
    if isinstance(value, Tensor):
        bound = math.sqrt(6 / ((1 + a**2) * fan))
        value.data.uniform_(-bound, bound)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            kaiming_uniform(v, fan, a)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            kaiming_uniform(v, fan, a)


def glorot(value: Any):
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)


def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()


def constant(value: Any, fill_value: float):
    if isinstance(value, Tensor):
        value.data.fill_(fill_value)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            constant(v, fill_value)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            constant(v, fill_value)


def zeros(value: Any):
    constant(value, 0.)


def ones(tensor: Any):
    constant(tensor, 1.)


def normal(value: Any, mean: float, std: float):
    if isinstance(value, Tensor):
        value.data.normal_(mean, std)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            normal(v, mean, std)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            normal(v, mean, std)


def reset(value: Any):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)




