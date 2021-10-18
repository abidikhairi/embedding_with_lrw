import warnings
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from models import MLPTrainer
from utils import load_data


def main(args):
    embedding = args.embedding 
    dataset = args.dataset
    workers = args.workers
    train_size = args.train_size
    
    _, _, labels = load_data(dataset)
    num_classes = len(np.unique(labels))
    features = np.load(embedding)


    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=train_size)

    classifiers = {
        'Logistic Regression': LogisticRegression(multi_class='ovr', n_jobs=workers),
        'Naive Bayes': GaussianNB(),
        'Decision Trees': DecisionTreeClassifier(),
        'MLP': MLPTrainer(num_classes=num_classes, feature_size=128, hidden_size=32)
    }

    for name, classifier in classifiers.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            print('Experiment on {} with {:.2f} % training samples'.format(name, train_size * 100))
            
            model = classifier.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            acc_score = accuracy_score(y_test, y_pred)
            print("{} accuracy score: {:.4f} %".format(name, acc_score * 100))
            print('-'*120)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run baseline classifiers on dataset features')

    parser.add_argument('--dataset', type=str, required=True, help='Dataset: cora, fifa, arxiv or yelp')
    parser.add_argument('--embedding', type=str, required=True, help='Embedding file')
    parser.add_argument('--workers', type=int, required=True, help='Number of cpu workers')
    parser.add_argument('--train_size', type=float, default=0.8, required=False, help='Train size, Default: 80 %')

    args = parser.parse_args()

    main(args)
