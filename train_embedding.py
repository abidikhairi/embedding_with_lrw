import logging
import argparse
import numpy as np
from gensim.models import Word2Vec


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def main(args):
    walk_path = args.walk_path
    embed_size = args.embed_size
    window_size = args.window_size
    negative = args.negative
    workers = args.workers
    out_dir = args.out_dir
    epochs = args.epochs
    
    walks = np.load(walk_path).tolist()

    skipgram = Word2Vec(sentences=walks, vector_size=embed_size, window=window_size, epochs=epochs, negative=negative, sg=1, workers=workers, min_count=1)

    keys = list(map(int, skipgram.wv.index_to_key))
    keys.sort()

    vectors = [skipgram.wv[idx] for idx in keys]

    embeddings = np.array(vectors)

    np.save(out_dir, embeddings) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train embedding with skipgram')

    parser.add_argument('--walk_path', type=str, required=True, help='numpy file containing walks')
    parser.add_argument('--embed_size', type=int, default=128, help='embedding dimension. Default: 128')
    parser.add_argument('--window_size', type=int, default=5, help='skipgram window')
    parser.add_argument('--negative', type=int, default=5, help='number of negative samples')
    parser.add_argument('--epochs', type=int, default=1, help='skip-gram epochs')
    parser.add_argument('--workers', type=int, default=2, help='number of cpu workers')
    parser.add_argument('--out_dir', type=str, required=True, help='directory to store embedding')

    args = parser.parse_args()

    main(args)
