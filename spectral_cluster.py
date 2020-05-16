import os
import numpy as np
from os.path import abspath, exists
from scipy import sparse
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


class SpectralCluster:
    def __init__(self, k=2):
        self.k = k

    def make_adjacency_matrix(self):
        pass

    def make_graph_laplacian(self):
        pass

    def compute_eigenvectors(self):
        pass