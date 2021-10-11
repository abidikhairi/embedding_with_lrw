import numpy as np
import networkx as nx
from scipy.spatial.distance import cosine

from operator import itemgetter
from tqdm import tqdm

class LazyRandomWalk:
    
    def __init__(self, graph: nx.Graph, num_walks: int = 10, walk_length: int = 80, sigma: float = 0.2, alpha: float = 0.2, similarity = cosine) -> None:
        r"""
            Source: https://arxiv.org/abs/2008.03639
            section: III-A-3
        """

        self.graph = graph
        self.num_walks = num_walks # FIXME: Non probilistic remove if process is determinist
        self.walk_length = walk_length
        self.sigma = sigma
        self.alpha = alpha
        self.similarity = similarity

        self.process_graph()

    def process_graph(self):
        r"""
            Calculate transition probabilities, using node attributes and pre-defined 
            node similarity function
        """

        for node, data in tqdm(self.graph.nodes(data=True), desc='Processing Graph'):
            neighbors = self.graph.neighbors(node)
            feat_u = data['node_attr']
            w = []
            
            for neighbor in neighbors:
                feat_v = self.graph.nodes[neighbor]['node_attr']
                wij = np.exp(self.similarity(feat_u, feat_v) / 2 * (self.sigma ** 2))
                w.append(wij)
            di = sum(w)

            self.graph.nodes[node]['d'] = di

    def simulate_walks(self):
        walks = []

        for node in tqdm(self.graph.nodes(), desc='Generating Walks'):
            walks.append(self._walk(node))
        
        return walks
            

    def _walk(self, start):
        length = self.walk_length
        walk = [start]

        while len(walk) < length:
            current = walk[-1]
            P = []

            feat_u = self.graph.nodes[current]['node_attr']
            di = self.graph.nodes[current]['d']
            neighbors = self.graph.neighbors(current)

            for neighbor in neighbors:
                feat_v = self.graph.nodes[neighbor]['node_attr']
                
                wij = np.exp(self.similarity(feat_u, feat_v) / 2 * (self.sigma ** 2))

                P.append((neighbor, (self.alpha * wij) / di))

            P.append((current, (1 - self.alpha)))

            next = max(P, key=itemgetter(1))[0]
            
            walk.append(next)
        
        return walk

