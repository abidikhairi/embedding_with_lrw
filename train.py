import dgl
import numpy as np
from node2vec import Node2Vec
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier, LogisticRegression

from dgl.data import CoraGraphDataset, CiteseerGraphDataset, FraudYelpDataset
from utils import load_arxiv, load_yelp, load_cora, load_citeseer


def run_personalized_walks(walk_path):

    walks = np.load(walk_path).tolist()
    
    skipgram = Word2Vec(sentences=walks, vector_size=128, negative=5, window=8, sg=1, workers=6)
    
    keys = list(map(int, skipgram.wv.index_to_key))
    keys.sort()

    vectors = [skipgram.wv[key] for key in keys]

    return np.array(vectors)

def run_node2vec(graph):
    node2vec = Node2Vec(graph, dimensions=128, walk_length=80, num_walks=5, p=1, q=1, workers=6)

    word2vec = node2vec.fit(negative=5, window=8)

    keys = list(map(int, word2vec.wv.index_to_key))
    
    keys.sort()

    vectors = [word2vec.wv[key] for key in keys]

    return np.array(vectors)

def run_classifier(embeddings, labels, train_ratio=0.8):
    """
        used to test both embedding outputs
    """
    x_train, x_test, y_train, y_test = train_test_split(embeddings, labels, train_size=train_ratio)

    train_accuracies = [] 
    test_accuracies = []
    

    classifier = LogisticRegression(multi_class='ovr', n_jobs=7)
            
    for _ in range(5):
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



def main():
    graph, _, labels = load_arxiv()

    embeddings = run_personalized_walks(walk_path='temp/arxiv-lrw-walks.npy')

    print('----- Lazy Random Walks -----')
    run_classifier(embeddings, labels, train_ratio=0.8)

    print('----- Node2Vec Walks -----')
    embeddings = run_node2vec(graph)
    run_classifier(embeddings, labels, train_ratio=0.8)

if __name__ == '__main__':
    main()
