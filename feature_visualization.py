import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from utils import load_arxiv


def main():
    _, features, labels = load_arxiv()

    tsne = TSNE(n_components=2, n_jobs=4)

    projection = tsne.fit_transform(features)
    
    plt.figure(figsize=(14, 10))
    plt.scatter(projection[:, 0], projection[:, 0], c=labels, cmap='Dark2', s=100)
    plt.show()

if __name__ == '__main__':
    main()
