import random

from igraph import Graph


class StochasticGraphGenerator(object):
    def __init__(self):
        self.edges = set()

    def _generate_edges_from_subsets_and_probabilities(self, membership, alpa, beta):
        beta_edges = set()
        for i in xrange(len(membership)):
            for j in range(i + 1, len(membership)):
                if membership[i] == membership[j]:
                    self._generate_edge_by_probability(i, j, alpa)
                else:
                    e = self._generate_edge_by_probability(i, j, beta)
                    if e != (-1, -1):
                        beta_edges.add(e)
        return self.edges, beta_edges

    def _generate_edge_by_probability(self, x, y, probability):
        if random.uniform(0, 1) <= probability:
            self.edges.add((x, y))
            return (x, y)
        return (-1, -1)

    def generate(self, vertices_count, membership, alpa, beta):
        self.edges = set()
        edges, beta_edges = self._generate_edges_from_subsets_and_probabilities(membership, alpa, beta)
        alpa_edges = self.edges - beta_edges
        graph = Graph()
        graph.add_vertices(vertices_count)
        graph.add_edges(set(list(edges)))
        return graph, beta_edges, alpa_edges

    @staticmethod
    def generate_equal_size_membership(vertices_count, community_size):
        subsets = []
        set_id = -1
        for i in xrange(vertices_count):
            if i % community_size == 0:
                set_id += 1
            subsets.append(set_id)
        return subsets

    @staticmethod
    def generate_random_size_membership(vertices_count, min_size, max_size):
        subsets = []
        set_id = -1
        membership_size = -1
        for i in xrange(vertices_count):
            if i % membership_size == 0:
                set_id += 1
                membership_size = random.randrange(min_size, max_size)
            subsets.append(set_id)
        return subsets
