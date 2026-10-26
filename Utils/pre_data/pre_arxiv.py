# Build an OGB arxiv subgraph with an upper-year cutoff and a lower-year scoring mask.
# The returned graph can include older context nodes outside the requested scoring window.
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import scipy
import scipy.io
from sklearn.preprocessing import label_binarize
from ogb.nodeproppred import NodePropPredDataset

import pickle as pkl


# Minimal single-graph container; load_ogb_arxiv below returns an adapted OGB object.
class NCDataset(object):
    def __init__(self, name):

        self.name = name
        self.graph = {}
        self.label = None

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_nc_dataset(data_dir, dataname, year=2020):
    """ Loader for NCDataset
        Returns NCDataset
    """
    # For this loader, year must be a two-element [lower, upper] sequence, not an integer.
    if dataname == 'ogb-arxiv':
        dataset = load_ogb_arxiv(data_dir=data_dir, year_bound=year, proportion=1.0)
    else:
        raise ValueError('Invalid dataname')
    return dataset


# Sort node/degree pairs by their degree entry.
def take_second(element):
    return element[1]


# Keep nodes through the inclusive upper year, optionally retaining a top-degree fraction.
# test_mask identifies retained nodes in the open-lower, closed-upper year interval.
def load_ogb_arxiv(data_dir, year_bound, proportion=1.0):
    import ogb.nodeproppred

    dataset = ogb.nodeproppred.NodePropPredDataset(name='ogbn-arxiv', root=data_dir)
    graph = dataset.graph

    node_years = graph['node_year']
    n = node_years.shape[0]
    node_years = node_years.reshape(n)

    d = np.zeros(len(node_years))

    # Count incident edges whose two endpoints both satisfy the upper-year cutoff.
    edges = graph['edge_index']
    for i in range(edges.shape[1]):
        if node_years[edges[0][i]] <= year_bound[1] and node_years[edges[1][i]] <= year_bound[1]:
            d[edges[0][i]] += 1
            d[edges[1][i]] += 1

    nodes = []
    for i, year in enumerate(node_years):
        if year <= year_bound[1]:
            nodes.append([i, d[i]])

    # Degree sorting also determines the new node order when proportion equals 1.0.
    nodes.sort(key=take_second, reverse=True)

    nodes = nodes[: int(proportion * len(nodes))]

    result_edges = []
    result_features = []
    result_labels = []

    for node in nodes:
        result_features.append(graph['node_feat'][node[0]])
    result_features = np.array(result_features)

    # Map original OGB node IDs to consecutive indices in the retained subgraph.
    ids = {}
    for i, node in enumerate(nodes):
        ids[node[0]] = i

    # Retain only edges with both endpoints selected; preserve their original direction.
    for i in range(edges.shape[1]):
        if edges[0][i] in ids and edges[1][i] in ids:
            result_edges.append([ids[edges[0][i]], ids[edges[1][i]]])
    result_edges = np.array(result_edges).transpose(1, 0)

    # Reorder labels with the same node list used for features and edge remapping.
    result_labels = dataset.labels[[node[0] for node in nodes]]

    edge_index = torch.tensor(result_edges, dtype=torch.long)
    node_feat = torch.tensor(result_features, dtype=torch.float)
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': node_feat.size(0)}
    dataset.label = torch.tensor(result_labels)
    node_years_new = [node_years[node[0]] for node in nodes]
    # Upper-year filtering was already applied; this mask now enforces only the lower bound.
    dataset.test_mask = (torch.tensor(node_years_new) > year_bound[0])

    return dataset
