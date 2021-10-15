import dgl
import numpy as np
from scipy.spatial.distance import hamming, cosine, euclidean

from walker import LazyRandomWalk, RandomWalk
from utils import load_citeseer, load_cora, load_yelp



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

    