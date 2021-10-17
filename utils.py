import dgl
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, FraudYelpDataset

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