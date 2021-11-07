import argparse
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from utils import load_data

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def run_classifier(embeddings, labels, train_ratio = 0.8, workers = 1, runs = 1):
    """
        used to test both embedding outputs
    """
    x_train, x_test, y_train, y_test = train_test_split(embeddings, labels, train_size=train_ratio)

    train_accuracies = [] 
    test_accuracies = []
    

    classifier = LogisticRegression(multi_class='ovr', n_jobs=workers)
            
    for _ in range(runs):
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


def experiment(args):
    dataset = args.dataset
    emebdding_path = args.embedding
    workers = args.workers
    runs = args.runs 
    train_size = args.train_size 
    
    _, _, labels = load_data(dataset)
    embedding = np.load(emebdding_path)
    
    print('----- LRW + SkipGram + Logistic Regression -----')
    run_classifier(embedding, labels, train_ratio=train_size, workers=workers, runs=runs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run experiment with logistic regression classifier')

    parser.add_argument('--embedding', type=str, required=True, help='embedding file')
    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--train_size', type=float, required=False, default=0.8, help='train size. Default: 80%')
    parser.add_argument('--workers', type=int, required=False, default=3, help='number of cpu workers')
    parser.add_argument('--runs', type=int, required=False, default=1, help='number of classifier runs for each dataset')
    args = parser.parse_args()
    
    experiment(args)
