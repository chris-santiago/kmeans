import os
from os.path import abspath, exists

import numpy as np
import pandas as pd
import scipy.linalg
from matplotlib import pyplot as plt
from scipy import sparse
from sklearn.cluster import KMeans


def graph_from_file(filename: str) -> np.ndarray:
    with open(filename, 'r') as file:
        edges = [line.split() for line in file.readlines()]
        return np.array(edges).astype(int)


def nodes_from_file(filename: str) -> np.ndarray:
    with open(filename, 'r') as file:
        nodes = [line.split()[2] for line in file.readlines()]
        return np.array(nodes).astype(int)


class SpectralCluster:
    def __init__(self, graph: np.ndarray, nodes: np.ndarray, k: int = 2):
        self.k = k
        self.adjacency_matrix: np.ndarray = self.make_adjacency_matrix(graph, len(nodes))
        self.degree_matrix: np.ndarray = np.array(0)
        self.graph_laplacian: np.ndarray = np.array(0)
        self.eig_vecs: np.ndarray = np.array(0)

    @staticmethod
    def make_adjacency_matrix(graph, n):
        graph_rows = graph[:, 0] - 1
        graph_cols = graph[:, 1] - 1
        val = np.ones((graph.shape[0]))
        return sparse.coo_matrix((val, (graph_rows, graph_cols)), shape=(n, n)).toarray()

    def make_degree_matrix(self):
        self.degree_matrix = np.diag(np.sum(self.adjacency_matrix, axis=1))
        return self

    def make_graph_laplacian(self):
        self.graph_laplacian = self.degree_matrix - self.adjacency_matrix
        return self

    def compute_eigenvectors(self):
        eig_vals, eig_vecs = scipy.linalg.eig(self.graph_laplacian)
        self.eig_vecs = eig_vecs.real
        return eig_vals.real, eig_vecs.real

    def get_k_eigenvectors(self):
        eig_vals, eig_vecs = self.compute_eigenvectors()
        sort_idx = np.argsort(eig_vals)
        sorted_vecs = eig_vecs[:, sort_idx]
        return sorted_vecs[:, :self.k]

    def fit(self):
        self.make_degree_matrix().make_graph_laplacian()
        kmeans = KMeans(n_clusters=self.k)
        kmeans.fit(self.get_k_eigenvectors())
        return kmeans

    def plot(self, labels: np.ndarray) -> None:
        for cluster in labels:
            pass


edges_file = 'homework1/data/edges.txt'
nodes_file = 'homework1/data/nodes.txt'
nodes = nodes_from_file(nodes_file)
graph = graph_from_file(edges_file)
cluster = SpectralCluster(graph, nodes)
test = cluster.fit()

idx = test.labels_

_, eig_vec = cluster.compute_eigenvectors()
kmeans = KMeans(2)
kmeans.fit(eig_vec[:, :2])
