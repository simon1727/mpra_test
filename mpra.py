import os

import numpy as np
import pandas as pd
import yaml

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

__SEED__ = 42
__PATH__ = '/data/tuxm/project/MPRA-collection/data/mpra_test/'

class MPRA_Collection:

    def __init__(self, folder = __PATH__):
        self.paper_names = os.listdir(folder)
        self.papers = dict()
        self.dataset_names = dict()
        self.datasets = dict()
        for paper_name in self.paper_names:
            paper = MPRA_Paper(folder, paper_name)
            self.papers[paper_name] = paper
            self.dataset_names[paper_name] = paper.list_datasets()
            self.datasets[paper_name] = paper.datasets

    def list_papers(self):
        return self.paper_names
    
    def n_papers(self):
        return len(self.papers)
    
    def get_paper(self, paper_name):
        return self.papers[paper_name]
    
    def list_datasets(self):
        return self.dataset_names
    
    def n_datasets(self):
        return sum([len(dataset) for dataset in self.datasets()])
    
    def get_dataset(self, paper_name, dataset_name):
        return self.datasets[paper_name][dataset_name]
    
    def __len__(self):
        return self.n_datasets()

class MPRA_Paper:
    
    def __init__(self, folder, paper_name):
        self.paper_name = paper_name
        self.dataset_names = [file[:-4] for file in os.listdir(os.path.join(folder, paper_name)) if '.csv' in file]
        self.datasets = dict()
        for dataset_name in self.dataset_names:
            # dataset = MPRA_Dataset.load(folder, paper_name, dataset_name)
            dataset = MPRA_Dataset(folder, paper_name, dataset_name)
            self.datasets[dataset_name] = dataset

    def list_datasets(self):
        return self.dataset_names
    
    def n_datasets(self):
        return len(self.dataset_names)
    
    def get_dataset(self, dataset_name):
        return self.datasets[dataset_name]
    
    def __len__(self):
        return self.n_datasets()

def seqs_to_onehot(
    seqs,
    len_max = 64,
    len_div = 16,
):
    len_max = (len_max + len_div - 1) // len_div * len_div
    def to_len_max(seq):
        len_seq = len(seq)
        len_0 = (len_max - len_seq + 1) // 2
        len_1 = (len_max - len_seq) // 2
        return 'N' * len_0 + seq + 'N' * len_1 if len_seq < len_max else seq[-len_0:len_1]
    seqs = [to_len_max(seq) for seq in seqs]

    alphabet = 'ATGC'
    alphabet_unknown = 'NX'
    value_unknown = 1.0 / len(alphabet)
    def to_uint8(seq):
        return np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
    hash_table = np.zeros((np.iinfo(np.uint8).max + 1, len(alphabet)))
    hash_table[to_uint8(alphabet)] = np.eye(len(alphabet))
    hash_table[to_uint8(alphabet_unknown)] = value_unknown
    def seq_to_onehot(seq):
        return hash_table[to_uint8(seq)]
    seqs = [seq_to_onehot(seq) for seq in seqs]
    
    return np.stack(seqs)

def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
class MPRA_Dataset:
    
    def __init__(self, 
        folder, 
        paper_name, 
        dataset_name, 
        info = dict(), 
        data = pd.DataFrame(), 
    ):
        self.folder = folder
        self.paper_name = paper_name
        self.dataset_name = dataset_name
        
        self.info = info
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data.iloc[idx]
    
    @staticmethod
    def load(folder, paper_name, dataset_name):
        file = os.path.join(folder, paper_name, dataset_name)
        with open(file + '.yaml', 'r') as f:
            info = yaml.safe_load(f)
        data = pd.read_csv(file + '.csv')
        return MPRA_Dataset(folder, paper_name, dataset_name, info, data)

    def load(self):
        file = os.path.join(self.folder, self.paper_name, self.dataset_name)
        with open(file + '.yaml', 'r') as f:
            self.info = yaml.safe_load(f)
        self.data = pd.read_csv(file + '.csv')

    def save(self):
        mkdir(os.path.join(self.folder, self.paper_name))
        file = os.path.join(self.folder, self.paper_name, self.dataset_name)
        with open(file + '.yaml', 'w') as f:
            yaml.safe_dump(self.info, f)
        self.data.to_csv(file + '.csv', index = False)
    
    def print_info(self):
        print('==== ==== ==== ====')
        print('\n'.join([f'{key}: {value}' for key, value in self.info.items()]))
        print('==== ==== ==== ====')

    def list_Y_names(self):
        return [name for name in self.data.columns if name.startswith('Y: ')]
    
    def list_obsX_names(self):
        return [name for name in self.data.columns if name.startswith('obsX: ')]
    
    def list_obsY_names(self):
        return [name for name in self.data.columns if name.startswith('obsY: ')]

    def init_XYobs(self):
        # compute X, Y, obsX and obsY from data

        Y_names = self.list_Y_names()
        obsX_names = self.list_obsX_names()
        obsY_names = self.list_obsY_names()

        self._X = torch.Tensor(seqs_to_onehot(self.data['X'].values)).transpose(1, 2)
        self._Y = self.data[Y_names]
        self._obsX = self.data[obsX_names]
        self._obsY = self.data[obsY_names]

        self.view_XYobs()

    def view_XYobs(self, Y_names = [], obsX_names = [], obsY_names = []):
        # select Y, obsX and obsY from data

        Y_names = Y_names if Y_names else self.list_Y_names()
        Y_names = [name for name in Y_names if name in self.list_Y_names()]

        obsX_names = obsX_names if obsX_names else self.list_obsX_names()
        obsY_names = obsY_names if obsY_names else self.list_obsY_names()
        obsX_names = [name for name in obsX_names if name in self.list_obsX_names()]
        obsY_names = [name for name in obsY_names if name in self.list_obsY_names()]

        # TODO: support more complex mask
        mask = self._Y[Y_names].notna().all(axis=1)

        self.X = self._X[mask]
        self.Y = torch.Tensor(self._Y[mask][Y_names].values)
        self.obsX = self._obsX[mask][obsX_names]
        self.obsY = self._obsY[mask][obsY_names]

    def split_rand(self, fracs = [0.8, 0.1, 0.1], seed = __SEED__):
        # frac_sum = sum(fracs)
        # fracs = [frac / frac_sum for frac in fracs]

        n = self.X.size(0)
        idx = torch.randperm(n, generator = torch.Generator().manual_seed(seed))

        subsets = []
        n_prefix = 0
        for frac in fracs[1:]:
            n_frac = (int)(n * frac)
            # slice_frac = slice(n_prefix, n_prefix + n_frac)
            # subset = (self.X[slice_frac], self.Y[slice_frac], self.obsX.iloc[slice_frac], self.obsY.iloc[slice_frac])
            idx_frac = idx[n_prefix:n_prefix + n_frac]
            subset = (self.X[idx_frac], self.Y[idx_frac], self.obsX.iloc[idx_frac], self.obsY.iloc[idx_frac])
            subsets.append(subset)
            n_prefix += n_frac
        # slice_frac = slice(n_prefix, n)
        # subset = (self.X[slice_frac], self.Y[slice_frac], self.obsX.iloc[slice_frac], self.obsY.iloc[slice_frac])
        idx_frac = idx[n_prefix:]
        subset = (self.X[idx_frac], self.Y[idx_frac], self.obsX.iloc[idx_frac], self.obsY.iloc[idx_frac])

        subsets.insert(0, subset) # subsets.append(subset)
        
        return subsets
