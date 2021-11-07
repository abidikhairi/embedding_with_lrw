import argparse
import networkx as nx

from utils import load_data

def main(args):
    dataset = args.dataset
    path = args.path
    
    graph, _, _ = load_data(dataset)

    nx.write_edgelist(graph, path, data=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate edgelist file for a given graph')

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--path', type=str, required=True)

    args = parser.parse_args()

    main(args)