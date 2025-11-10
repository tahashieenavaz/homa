import torch
import numpy
from ..Dataset import Dataset


class CoraDataset(Dataset):
    def __init__(self):
        pass

    def load(path: str):
        idx_features_labels = numpy.genfromtxt(f"{path}/cora.content", dtype=str)
        features = torch.tensor(idx_features_labels[:, 1:-1].astype(numpy.float32))
        classes = {label: i for i, label in enumerate(set(idx_features_labels[:, -1]))}
        labels = torch.tensor(
            [classes[label] for label in idx_features_labels[:, -1]], dtype=torch.long
        )

        # map node ids to index
        ids = idx_features_labels[:, 0].astype(numpy.int32)
        id_map = {j: i for i, j in enumerate(ids)}

        # read edges
        edges_unordered = numpy.genfromtxt(f"{path}/cora.cites", dtype=numpy.int32)
        edges = numpy.array(
            list(map(lambda x: [id_map[x[0]], id_map[x[1]]], edges_unordered))
        )

        # build adjacency matrix
        n = labels.shape[0]
        adj = torch.zeros((n, n))
        for i, j in edges:
            adj[i, j] = 1
            adj[j, i] = 1

        # normalize features
        features = features / features.sum(1, keepdim=True).clamp(min=1)

        # simple splits
        idx_train = torch.arange(140)
        idx_val = torch.arange(200, 500)
        idx_test = torch.arange(500, 1500)

        return features, adj, labels, idx_train, idx_val, idx_test
