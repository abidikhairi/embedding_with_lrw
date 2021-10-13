import dgl
import numpy as np
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, FraudYelpDataset
from scipy.spatial.distance import hamming, cosine, euclidean

from walker import LazyRandomWalk, RandomWalk


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

def main():
    graph, features = load_yelp()

    for node, _ in graph.nodes(data=True):
        if hasattr(features[node], 'numpy'):
            graph.nodes[node]['node_attr'] = features[node].numpy()
        else:
            graph.nodes[node]['node_attr'] = features[node]

    walks = LazyRandomWalk(graph, similarity=[cosine]).simulate_walks()

    np_walks = np.array(walks)

    np.save('temp/yelp-lrw-walks', np_walks)

if __name__ == '__main__':
    main()

    