import argparse
import numpy as np
from tqdm import tqdm

from walker import RandomWalk
from utils import load_data


def main(args):
    dataset_name = args.dataset
    num_walks = args.num_walks
    walk_length = args.walk_length
    
    
    graph, features, _ = load_data(dataset_name)

    for node, _ in tqdm(graph.nodes(data=True), desc='Init Node Features'):
        if hasattr(features[node], 'numpy'):
            graph.nodes[node]['node_attr'] = features[node].numpy()
        else:
            graph.nodes[node]['node_attr'] = features[node]
    
    walks = RandomWalk(graph, num_walks=num_walks, walk_length=walk_length).simulate_walks()

    np_walks = np.array(walks)

    np.save('temp/{}-deepwalk-walks'.format(dataset_name), np_walks)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='dataset to use: arxiv, cora, yelp, citeseer')
    parser.add_argument('--num_walks', default=5, type=int, required=False, help='Number of walks starting from each node')
    parser.add_argument('--walk_length', default=80, type=int, required=False, help='Walk Length for each node')
    args = parser.parse_args()

    main(args)