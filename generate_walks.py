import numpy as np
from dgl.data import CoraGraphDataset
from scipy.spatial.distance import hamming

from walker import LazyRandomWalk


def load_cora():
    dataset = CoraGraphDataset(verbose=False)

    features = dataset[0].ndata['feat']
    graph = dataset[0].to_networkx().to_undirected()

    return graph, features


def main():
    graph, features = load_cora()

    for node, _ in graph.nodes(data=True):
        graph.nodes[node]['node_attr'] = features[node].numpy()

    walks = LazyRandomWalk(graph, similarity=hamming).simulate_walks()

    np_walks = np.array(walks)

    np.save('temp/lrw-walks.npx', np_walks)

if __name__ == '__main__':
    main()

    