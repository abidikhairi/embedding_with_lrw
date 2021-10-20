import os
import dgl
import numpy as np
import pandas as pd
import networkx as nx
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, FraudYelpDataset


def normalize(x):
    """normalize a set of vectors to have unit length

    Args:
        x (np.ndarray): array or matrix

    Returns:
        np.ndarray: normalized array or matrix
    """    
    norm = x / np.linalg.norm(x)
    return norm

def load_fifa():
    datapath = os.environ['DATA_PATH']
    fifa_dir = os.path.join(datapath, 'fifa', 'processed')
    
    nation_edges_file = os.path.join(fifa_dir, 'player_nation_player.csv')
    club_edges_file = os.path.join(fifa_dir, 'player_club_player.csv')
    features_file = os.path.join(fifa_dir, 'fifa21_features.npy')
    labels_file = os.path.join(fifa_dir, 'fifa21_labels.npy')

    features = np.load(features_file)
    labels = np.load(labels_file)
    nation_edges = pd.read_csv(nation_edges_file)
    club_edges = pd.read_csv(club_edges_file)
    
    graph = nx.Graph()

    graph.add_edges_from(nation_edges.values)
    graph.add_edges_from(club_edges.values)

    return graph, features, labels

def get_similarity_func(name):
    import scipy.spatial.distance as distance
    if name == 'cosine':
        return distance.cosine
    elif name == 'euclidean':
        return distance.euclidean
    elif name == 'hamming':
        return distance.hamming
    elif name == 'taxi':
        return distance.minkowski

def load_data(name):
    if name == 'arxiv':
        return load_arxiv()
    elif name == 'cora':
        return load_cora()
    elif name == 'citeseer':
        return load_citeseer()
    elif name == 'yelp':
        return load_yelp()
    elif name == 'fifa':
        return load_fifa()

def load_arxiv():
    dataset = DglNodePropPredDataset(name='ogbn-arxiv', root='temp')

    graph, labels = dataset[0]

    features = graph.ndata['feat'].numpy()
    labels = labels.flatten().numpy()
    graph = graph.to_networkx()

    return graph, features, labels

def load_cora():
    dataset = CoraGraphDataset(verbose=False)

    features = dataset[0].ndata['feat'].numpy()
    labels = dataset[0].ndata['label'].numpy()
    graph = dataset[0].to_networkx().to_undirected()

    return graph, features, labels

def load_citeseer():
    dataset = CiteseerGraphDataset(verbose=False)

    features = dataset[0].ndata['feat'].numpy()
    graph = dataset[0].to_networkx().to_undirected()
    labels = graph.ndata['label'].numpy()

    return graph, features, labels

def load_yelp():
    dataset = FraudYelpDataset()
    
    graph = dgl.to_homogeneous(dataset[0]).to_networkx().to_undirected()
    features = dataset[0].ndata['feature'].numpy()
    labels = graph.ndata['label'].numpy()

    return graph, features, labels