import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from utils import load_data

def get_model(name, workers):
    if name == 'pca':
        return PCA(n_components=2)
    elif name == 'tsne':
        return TSNE(n_components=2, n_jobs=workers)

def main(args):
    title = args.title
    _, _, labels = load_data(args.dataset)
    embedding = np.load(args.embedding)
    reduction = get_model(args.algorithm, args.workers)

    projection = reduction.fit_transform(embedding, reduction)
    
    plt.figure(figsize=(14, 10))
    plt.title(title)
    plt.scatter(projection[:, 0], projection[:, 1], c=labels, cmap='Dark2', s=10)
    plt.savefig('figures/{}'.format(title.lower().replace(' ', '')))
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', type=str, required=True, help='Figure title')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--embedding', required=True, help='Embedding file')
    parser.add_argument('--algorithm', required=False, help='Dimensionaly reduction method', choices=['tsne', 'pca'], default='tsne')
    parser.add_argument('--workers', type=int, required=False, help='Number of cpu workers', default=1)
    args = parser.parse_args()

    main(args)
