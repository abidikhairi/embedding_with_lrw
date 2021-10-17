import numpy as np
import networkx as nx
from scipy.spatial.distance import cosine
from scipy import sparse
from scipy.sparse import csr_matrix

from tqdm import tqdm


class RandomWalk:
    def __init__(self, graph: nx.Graph, num_walks: int = 10, walk_length: int = 80) -> None:
        r"""
            Generate randomly uniform random walks
        """
        self.graph = graph
        self.num_walks = num_walks
        self.walk_length = walk_length

    def simulate_walks(self):
        walks = []

        for _ in tqdm(range(self.num_walks), desc='Generating Walks'):
            for node in self.graph.nodes():
                walks.append(self._walk(node))
        
        return walks


    def _walk(self, start):
        length = self.walk_length
        walk = [start]

        while len(walk) < length:
            current = walk[-1]

            neighbors = list(self.graph.neighbors(current))
            
            next = np.random.choice(neighbors)
            walk.append(next)
            
        return walk

class LazyRandomWalk:
    
    def __init__(self, graph: nx.Graph, num_walks: int = 5, walk_length: int = 80, sigma: float = 0.2, alpha: float = 0.2, similarity = cosine) -> None:
        r"""
            Source: https://arxiv.org/abs/2008.03639
            section: III-A-3
        """

        self.graph = graph
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.sigma = sigma
        self.alpha = alpha
        self.similarity = similarity
        self.transition = None
        self.nodes = np.arange(self.graph.number_of_nodes())
        
        self.process_graph()

    def weighting(self, u, v):
        sigma = self.sigma

        xu = self.graph.nodes[u]['node_attr']
        xv = self.graph.nodes[v]['node_attr']
        
        return np.exp(- self.similarity(xu, xv) / 2 * (sigma ** 2))

    def process_graph(self):
        r"""
            Calculate transition probabilities, using node attributes and pre-defined 
            node-wise similarity function
        """
        alpha = self.alpha
        adj = nx.adjacency_matrix(self.graph)
        edges = np.stack(adj.nonzero()).T.tolist()
        W = sparse.lil_matrix((self.graph.number_of_nodes(), self.graph.number_of_nodes()))

        for u, v in tqdm(edges, desc='Computing Transition probabilities'):
            score = self.weighting(u, v)
            W[u, v] = score

        rows = self.nodes
        cols = self.nodes

        alphas = np.ones(self.graph.number_of_nodes()) * (1 - alpha)
        degress = 1./ np.array(list(dict(self.graph.degree()).values()))       
        A = sparse.coo_matrix((alphas, (rows, cols)))
        D = sparse.coo_matrix((degress, (rows, cols)))

        P = (W + A) @ D

        self.transition = P

    def simulate_walks(self):
        walks = []

        for _ in range(self.num_walks):
            for node in tqdm(self.graph.nodes(), desc='Generating Walks'):
                walks.append(self._walk(node))
        
        return walks
            

    def _walk(self, start):
        length = self.walk_length
        walk = [start]

        while len(walk) < length:
            current = walk[-1]
            probs = self.transition[current, :]
            
            probs = probs.todense() 
            probs /= probs.sum() # normalize transition # TODO: try normalize by node degree
            probs = np.array(probs)[0]

            next = np.random.choice(self.nodes, p=probs)

            walk.append(next)
        
        return walk

