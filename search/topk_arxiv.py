import argparse
import torch
import numpy as np
import pandas as pd

import torch.nn.functional as F


def main(args):
    embedding_file = args.embedding_file
    k = args.k
    paper = args.paper
    titles_file = args.titles_file
    method = args.method

    df = pd.read_csv(titles_file)
    embeddings = torch.from_numpy(np.load(embedding_file))

    x = embeddings[paper]

    xs = x.repeat(len(embeddings), 1)
    
    distances  = F.cosine_similarity(xs, embeddings)
    scores, indices = torch.topk(distances, k=k + 1)
    
    sim_df = df.iloc[indices.tolist(), 2].reset_index()
    sim_df['scores'] = scores.numpy()

    row = sim_df.iloc[0, :]
    print('Query: {}'.format(row[1]))
    print('\n')
    print('Most similar papers (methdo: {})'.format(method))
    print('-'*100)
    for tup in sim_df.iloc[1:,:].itertuples():
        print('|{}, {}, {}|'.format(tup[1], tup[2], tup[3])) 
        print('-'*100)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arxiv dataset similarity search')

    parser.add_argument('--k', type=int, required=False, default=5, help='Top K. Defaults: 5.')
    parser.add_argument('--embedding_file', type=str, required=True, help='embedding file')
    parser.add_argument('--paper', type=int, help='paper index')
    parser.add_argument('--titles_file', type=str, required=True, help='path to arxiv_titles.csv file')
    parser.add_argument('--method', type=str, required=True, help='method used to generate embedding: (lrw or node2vec)', choices=['lrw', 'node2vec'])
    
    args = parser.parse_args()
    
    main(args)
