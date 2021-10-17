import os
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

from utils import load_data

def main(args):
    emb_file = args.embedding
    num_clusters = args.num_clusters
    workers = args.workers
    dataset = args.dataset
    method = args.method
    
    # n_jobs is deprecated 
    os.environ['OMP_NUM_THREADS'] = workers

    _, _, true_labels = load_data(dataset)

    features = np.load(emb_file)
    
    amis = []
    nmis = []
    
    for _ in range(10):
        kmeans = KMeans(n_clusters=num_clusters)
        model = kmeans.fit(features)
        labels = model.labels_
        amis.append(
            adjusted_mutual_info_score(true_labels, labels)
        )
        nmis.append(
            normalized_mutual_info_score(true_labels, labels)
        )
        
    print("{} Dataset: {} method".format(dataset.capitalize(), method))
    print("AMI: ", np.mean(amis))
    print("NMI: ", np.mean(nmis))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True)
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--embedding', type=str, required=True, help='Embedding file')
    parser.add_argument('--num_clusters', type=int, required=True, help='Number of clusters')
    parser.add_argument('--workers', type=str, required=False, help='Number of cpu workers', default='1')

    args = parser.parse_args()

    main(args)
