import dgl
import numpy as np
from scipy.spatial.distance import hamming, cosine, euclidean

from walker import LazyRandomWalk, RandomWalk
from utils import load_citeseer, load_cora, load_yelp, load_arxiv


def main():
    graph, features, _ = load_arxiv()

    for node, _ in graph.nodes(data=True):
        if hasattr(features[node], 'numpy'):
            graph.nodes[node]['node_attr'] = features[node].numpy()
        else:
            graph.nodes[node]['node_attr'] = features[node]

    walks = LazyRandomWalk(graph, similarity=[cosine]).simulate_walks()

    np_walks = np.array(walks)

    np.save('temp/arxiv-lrw-walks', np_walks)

if __name__ == '__main__':
    main()

    