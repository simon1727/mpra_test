"""Main module."""

import yaml
import pandas as pd
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

from .utils import *

__PATH__ = '/data/tuxm/project/MPRA-collection/data/mpra_test/'

class MPRA_Dataset:
    
    def __init__(self, 
        folder = __PATH__, name_paper = '', name_dataset = '', 
        info = dict(), data = pd.DataFrame(), 
        X = pd.DataFrame(), Y = pd.DataFrame(), obs_X = pd.DataFrame(), obs_Y = pd.DataFrame(), 
    ):
        self.folder = folder
        self.name_paper = name_paper
        self.name_dataset = name_dataset
        
        self.info = info
        self.data = data if data.shape != (0, 0) else XYobs_to_data(X, Y, obs_X, obs_Y)
        self.X, self.Y, self.obs_X, self.obs_Y = data_to_XYobs(self.data)

    def __len__(self):
        return self.data.shape[0]
    
    @property
    def shape(self):
        return self.data.shape

    # IO-related
    @staticmethod
    def load(name_paper: str, name_dataset: str, folder = __PATH__):
        with open(os.path.join(folder, name_paper, name_dataset + '.yaml'), 'r') as file:
            info = yaml.safe_load(file)
        data = pd.read_csv(os.path.join(folder, name_paper, name_dataset + '.csv'))
        return MPRA_Dataset(folder, name_paper, name_dataset, info, data)

    def reload(self):
        file = os.path.join(self.folder, self.name_paper, self.name_dataset)
        with open(file + '.yaml', 'r') as f:
            self.info = yaml.safe_load(f)
        self.data = pd.read_csv(file + '.csv')
        self.X, self.Y, self.obs_X, self.obs_Y = data_to_XYobs(self.data)

    def save(self):
        mkdir(os.path.join(self.folder, self.name_paper))
        file = os.path.join(self.folder, self.name_paper, self.name_dataset)
        with open(file + '.yaml', 'w') as f:
            yaml.safe_dump(self.info, f)
        self.data.to_csv(file + '.csv', index = False)
 
    # PyTorch-related
    def to_Dataset(self, cols_Y: list = []):
        cols_Y = cols_Y if cols_Y else [col for col in self.data.columns if col.startswith('Y: ')]
        cols_Y = [col[3:] if col.startswith('Y: ') else col for col in cols_Y]
        mask = self.Y[cols_Y].notna().all(axis = 1)
        len_max = self.X['X'][mask].str.len().max()
        _X = torch.Tensor(seqs_to_onehot(self.X['X'][mask].values, len_max=len_max)).transpose(1, 2)
        _Y = torch.Tensor(self.Y[cols_Y][mask].values)
        return TensorDataset(_X, _Y)

    def to_DataLoader(self, 
        batch_size: int, 
        num_workers: int = 1, 
        shuffle: bool = True, 
        *args, **kwargs,
    ):
        return DataLoader(self.to_Dataset(*args, **kwargs), 
            batch_size = batch_size, 
            num_workers = num_workers, 
            shuffle = shuffle,
        )

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.data.iloc[index]
        elif isinstance(index, str):
            return self.data[index]
        elif isinstance(index, slice):
            return MPRA_Dataset(
                self.folder, self.name_paper, self.name_dataset, 
                self.info, self.data.iloc[index], 
            )
        elif isinstance(index, pd.Series):
            return MPRA_Dataset(
                self.folder, self.name_paper, self.name_dataset, 
                self.info, self.data[index], 
            )
        elif isinstance(index, np.ndarray):
            return MPRA_Dataset(
                self.folder, self.name_paper, self.name_dataset, 
                self.info, self.data.iloc[index], 
            )
        elif isinstance(index, torch.Tensor):
            index = index.tolist()
            return MPRA_Dataset(
                self.folder, self.name_paper, self.name_dataset, 
                self.info, self.data.iloc[index], 
            )
        elif isinstance(index, list):
            if all(isinstance(i, int) for i in index):
                return MPRA_Dataset(
                    self.folder, self.name_paper, self.name_dataset, 
                    self.info, self.data.iloc[index], 
                )
            elif all(isinstance(i, str) for i in index):
                return MPRA_Dataset(
                    self.folder, self.name_paper, self.name_dataset, 
                    self.info, self.data[index], 
                )
            else:
                raise TypeError(f'List of distinct index: {set(i.__class__ for i in index)}')
        else:
            raise TypeError(f'Unsupported index: {index.__class__}')

    @property
    def seq(self):
        return self.X
    @seq.setter
    def seq(self, value):
        self.X = value
    @property
    def obs_seq(self):
        return self.obs_X
    @obs_seq.setter
    def obs_seq(self, value):
        self.obs_X = value
    
    @property
    def readout(self):
        return self.Y
    @readout.setter
    def readout(self, value):
        self.Y = value
    @property
    def obs_readout(self):
        return self.obs_Y
    @obs_readout.setter
    def obs_readout(self, value):
        self.obs_Y = value
