import os
import numpy as np
import pandas as pd
import scipy.linalg
from os.path import abspath, exists
from scipy import sparse
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


class SpectralCluster:
    def __init__(self, k: int = 2):
        self.k = k
        self.A: np.ndarray = np.array(0)
        self.D: np.ndarray = np.array(0)
        self.L: np.ndarray = np.array(0)
        self.eig_vecs: np.ndarray = np.array(0)

    def from_graph(self, graph, n):
        graph_rows = graph[:, 0] - 1
        graph_cols = graph[:, 1] - 1
        val = np.ones((graph.shape[0]))
        return sparse.coo_matrix((val, (graph_rows, graph_cols)), shape=(n, n)).toarray()

    def make_adjacency_matrix(self):
        pass

    def make_degree_matrix(self):
        self.D = np.diag(np.sum(self.A, axis=1))
        return self

    def make_graph_laplacian(self):
        self.L = self.D - self.A
        return self

    def compute_eigenvectors(self):
        eig_vals, self.eig_vecs = scipy.linalg.eig(self.L)
        return self

    def fit(self):
        pass

    def plot(self):
        pass


def main():
    nodes = pd.read_table('homework1/data/nodes.txt')