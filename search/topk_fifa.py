import argparse
import torch
import numpy as np
import pandas as pd

import torch.nn.functional as F

def main(args):
    embedding_file = args.embedding_file
    k = args.k
    player = args.player
    players_file = args.players_file

    df = pd.read_csv(players_file)
    embeddings = torch.from_numpy(np.load(embedding_file))

    x = embeddings[player]

    xs = x.repeat(len(embeddings), 1)
    
    distances  = F.cosine_similarity(xs, embeddings)
    scores, indices = torch.topk(distances, k=k + 1)
    
    sim_df = df.iloc[indices.tolist(), 2].reset_index()
    sim_df['scores'] = scores.numpy()
    
    print(sim_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fifa dataset similarity search')

    parser.add_argument('--k', type=int, required=False, default=5, help='Top K. Defaults: 5.')
    parser.add_argument('--embedding_file', type=str, required=True, help='embedding file')
    parser.add_argument('--player', type=int, help='player index')
    parser.add_argument('--players_file', type=str, required=True, help='path to players.csv file')
    parser.add_argument('--method', type=str, required=True, help='method used to generate embedding: (lrw or node2vec)', choices=['lrw', 'node2vec'])
    
    args = parser.parse_args()
    
    main(args)
