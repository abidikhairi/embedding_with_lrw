import numpy as np
from dgl.data import CoraGraphDataset, CiteseerGraphDataset
from scipy.spatial.distance import hamming, cosine

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

def main():
    graph, features = load_citeseer()

    for node, _ in graph.nodes(data=True):
        graph.nodes[node]['node_attr'] = features[node].numpy()

    walks = LazyRandomWalk(graph, similarity=hamming).simulate_walks()

    np_walks = np.array(walks)

    np.save('temp/citeseer-lrw-walks', np_walks)

    walks = RandomWalk(graph).simulate_walks()
    np_walks = np.array(walks)

    np.save('temp/citeseer-random-walks', np_walks)

if __name__ == '__main__':
    main()

    