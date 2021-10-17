import argparse
import logging
import numpy as np
from node2vec import Node2Vec
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from utils import load_arxiv, load_yelp, load_cora, load_citeseer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def run_skipgram(walk_path):

    walks = np.load(walk_path).tolist()

    skipgram = Word2Vec(sentences=walks, vector_size=128, negative=5, window=8, sg=1, workers=6, epochs=10)
    
    keys = list(map(int, skipgram.wv.index_to_key))
    keys.sort()
    
    vectors = [skipgram.wv[key] for key in keys]

    return np.array(vectors)

def run_node2vec(graph):
    node2vec = Node2Vec(graph, dimensions=128, walk_length=80, num_walks=5, p=1, q=1, workers=6)

    word2vec = node2vec.fit(negative=5, window=8, epochs=1)

    keys = list(map(int, word2vec.wv.index_to_key))
    
    keys.sort()

    vectors = [word2vec.wv[key] for key in keys]
    import pdb; pdb.set_trace()
    return np.array(vectors)

def run_classifier(embeddings, labels, train_ratio=0.8):
    """
        used to test both embedding outputs
    """
    x_train, x_test, y_train, y_test = train_test_split(embeddings, labels, train_size=train_ratio)

    train_accuracies = [] 
    test_accuracies = []
    

    classifier = LogisticRegression(multi_class='ovr', n_jobs=7)
            
    for _ in range(1):
        classifier = classifier.fit(x_train, y_train)
                
        train_acc = accuracy_score(y_train, classifier.predict(x_train))
        test_acc = accuracy_score(y_test, classifier.predict(x_test))

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    print('-'*120)
    print(f'OneVsRest Classifier')
    print(f'\t\t\t: Train Size: {train_ratio * 100} %')
    print('\t\t\t: Train Accuracy: {:.4f} %'.format(np.mean(train_accuracies) * 100))
    print('\t\t\t: Test Accuracy: {:.4f} %'.format(np.mean(test_accuracies) * 100))
    print('-'*120)

def train():
    graph, _, _ = load_arxiv()

    embeddings = run_skipgram(walk_path='temp/arxiv-lrw-walks.npy')
    np.save('temp/embeddings/arxiv-lrw', embeddings)

    embeddings = run_node2vec(graph)
    np.save('temp/embeddings/arxiv-node2vec', embeddings)

def experiment():
    _, _, labels = load_arxiv()
    lrw_emebdding = np.load('temp/embeddings/arxiv-lrw.npy')
    node2vec_emebdding = np.load('temp/embeddings/arxiv-node2vec.npy')

    print('----- LRW + SkipGram + Logistic Regression -----')
    run_classifier(lrw_emebdding, labels, train_ratio=0.8)

    print('----- Node2Vec + SkipGram + Logistic Regression -----')
    run_classifier(node2vec_emebdding, labels, train_ratio=0.8)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='train', help='Task to run, experiment or train', choices=['train', 'experiment'], required=True)
    args = parser.parse_args()

    if args.method == 'train':
        train()
    else:
        experiment()
