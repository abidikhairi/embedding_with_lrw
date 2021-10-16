import argparse
import numpy as np
from tqdm import tqdm

from walker import LazyRandomWalk
from utils import load_data, get_similarity_func


def main(args):
    dataset_name = args.dataset
    similarity_func = get_similarity_func(args.similarity)
    num_walks = args.num_walks
    walk_length = args.walk_length
    
    
    graph, features, _ = load_data(dataset_name)

    for node, _ in tqdm(graph.nodes(data=True), desc='Init Node Features'):
        if hasattr(features[node], 'numpy'):
            graph.nodes[node]['node_attr'] = features[node].numpy()
        else:
            graph.nodes[node]['node_attr'] = features[node]
    
    walks = LazyRandomWalk(graph, num_walks=num_walks, walk_length=walk_length, similarity=similarity_func).simulate_walks()

    np_walks = np.array(walks)

    np.save('temp/{}-lrw-walks'.format(dataset_name), np_walks)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='dataset to use: arxiv, cora, yelp, citeseer')
    parser.add_argument('--num_walks', default=5, type=int, required=False, help='Number of walks starting from each node')
    parser.add_argument('--walk_length', default=80, type=int, required=False, help='Walk Length for each node')
    parser.add_argument('--similarity', type=str, required=True, help='Similarity function to use e.g cosine, hamming, euclidean')
    args = parser.parse_args()

    main(args)

    