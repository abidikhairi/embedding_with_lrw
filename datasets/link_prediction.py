import numpy as np
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class LinkPredictionDataset(object):

    def __init__(self, graph: nx.Graph, embeddings: np.ndarray, negative: int = 5, operator: str  = 'hadamard') -> None:
        """create edges dataset

        Args:
            graph (nx.Graph): graph
            embeddings (np.ndarray): embedding table
            negative (int, optional): number of negative edges per positive edge. Defaults to 5.
            operator (str, optional): operator to aggregate node features and construct edge feature. Defaults to hadamard.
        """
        super().__init__()
        self.negative = negative
        self.embeddings = embeddings
        self.operator = np.multiply if operator == 'hadamard' else np.add
        self.num_nodes = graph.number_of_nodes()
        self.num_edges = graph.number_of_edges()
        self.edges = list(graph.edges())

    def __len__(self):
        """return number of edges

        Returns:
            int: number of training edges
        """        
        return self.num_edges 

    def __getitem__(self, index):
        pos_edge = self.edges[index]
        features = list(map(lambda n: self.embeddings[n], pos_edge))
        pos_feature = (self.operator(*features), 1)
        neg_features = []
        num_negatives = 0
        
        while num_negatives < self.negative:
            u = np.random.randint(0, self.num_nodes)
            v = np.random.randint(0, self.num_nodes)
            if (u, v) in self.edges:
                continue

            features = list(map(lambda n: self.embeddings[n], pos_edge))
            edge_feature = self.operator(*features)
            num_negatives += 1
            neg_features.append((edge_feature, 0))
        
        batch = [item for item in neg_features]
        batch.append(pos_feature)

        features = list(map(lambda tup: tup[0], batch))
        labels = list(map(lambda tup: tup[1], batch))
        
        return features, labels

    def generate_dataset(self, train_size=0.8):
        x_data, y_data = [], []

        for index, _ in enumerate(self.edges):
            features, labels = self.__getitem__(index)
            x_data.extend(features)
            y_data.extend(labels)

        x_data, y_data = shuffle(x_data, y_data)

        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=train_size)
        
        return x_train, x_test, y_train, y_test