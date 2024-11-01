import numpy as np
import torch

import scipy.sparse as sp
from itertools import repeat
import os.path as osp
import warnings
import sys
from torch_geometric.data import InMemoryDataset, Data


class Dpr2Pyg(InMemoryDataset):
    """Convert deeprobust data (sparse matrix) to pytorch geometric data (tensor, edge_index)

    Parameters
    ----------
    dpr_data :
        data instance of class from deeprobust.graph.data, e.g., deeprobust.graph.data.Dataset,
        deeprobust.graph.data.PtbDataset, deeprobust.graph.data.PrePtbDataset
    transform :
        A function/transform that takes in an object and returns a transformed version.
        The data object will be transformed before every access. For example, you can
        use torch_geometric.transforms.NormalizeFeatures()

    Examples
    --------
    We can first create an instance of the Dataset class and convert it to
    pytorch geometric data format.

    >>> from deeprobust.graph.data import Dataset, Dpr2Pyg
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> pyg_data = Dpr2Pyg(data)
    >>> print(pyg_data)
    >>> print(pyg_data[0])
    """

    def __init__(self, adj, features, labels, idx_train, idx_val, idx_test, transform=None, **kwargs):
        root = 'data/'  # dummy root; does not mean anything
        self.adj = adj
        self.features = features
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        super(Dpr2Pyg, self).__init__(root, transform)
        pyg_data = self.process()
        self.data, self.slices = self.collate([pyg_data])
        self.transform = transform

    def process(self):
        edge_index = torch.LongTensor(self.adj.nonzero())
        # by default, the features in pyg data is dense
        if sp.issparse(self.features):
            x = torch.FloatTensor(self.features.todense()).float()
        else:
            x = torch.FloatTensor(self.features).float()
        y = torch.LongTensor(self.labels)
        idx_train, idx_val, idx_test = self.idx_train, self.idx_val, self.idx_test
        data = Data(x=x, edge_index=edge_index, y=y)
        train_mask = index_to_mask(idx_train, size=y.size(0))
        val_mask = index_to_mask(idx_val, size=y.size(0))
        test_mask = index_to_mask(idx_test, size=y.size(0))
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        return data

    def update_edge_index(self, adj):
        """ This is an inplace operation to substitute the original edge_index
        with adj.nonzero()

        Parameters
        ----------
        adj: sp.csr_matrix
            update the original adjacency into adj (by change edge_index)
        """
        self.data.edge_index = torch.LongTensor(adj.nonzero())
        self.data, self.slices = self.collate([self.data])

    def get(self, idx):
        if self.slices is None:
            return self.data
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[self.data.__cat_dim__(key, item)] = slice(slices[idx],
                                                        slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def _download(self):
        pass


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask