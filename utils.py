from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, FraudYelpDataset

def load_arxiv():
    dataset = DglNodePropPredDataset(name='ogbn-arxiv', root='temp')

    graph, labels = dataset[0]

    features = graph.ndata['feat'].numpy()
    labels = labels.flatten().numpy()
    graph = graph.to_networkx()

    return graph, features, labels

def load_cora():
    dataset = CoraGraphDataset(verbose=False)

    features = dataset[0].ndata['feat']
    graph = dataset[0].to_networkx().to_undirected()

    return graph, features

def load_citeseer():
    dataset = CiteseerGraphDataset(verbose=False)

    features = dataset[0].ndata['feat']
    graph = dataset[0].to_networkx().to_undirected()

    return graph, features

def load_yelp():
    dataset = FraudYelpDataset()
    
    graph = dgl.to_homogeneous(dataset[0]).to_networkx().to_undirected()
    features = dataset[0].ndata['feature'].numpy()

    return graph, features