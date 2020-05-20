"""Module for basic Spectral Clustering"""
from typing import Tuple, Optional

import numpy as np
import scipy.linalg
from sklearn.cluster import KMeans


def edges_from_file(filename: str) -> np.ndarray:
    """
    Converts a text file of edges into a numpy array.
    Expected file format:
        14	312
        14	1152
        14	1461
        14	726

    :param filename: String filename.
    :return: A numpy array (n, 2) of edges.
    """
    with open(filename, 'r') as file:
        return np.array([line.split() for line in file.readlines()]).astype(int)


def nodes_from_file(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a text file of nodes into a numpy array.
    Expected format:
        1	"100monkeystyping.com"	0	"Blogarama"
        2	"12thharmonic.com/wordpress"	0	"BlogCatalog"
        3	"40ozblog.blogspot.com"	0	"Blogarama,BlogCatalog"
        4	"4lina.tblog.com"	0	"Blogarama"

    :param filename: String filename.
    :return: A tuple of nodes and respective labels.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
        nodes = np.array([line.split()[0] for line in lines]).astype(int)
        labels = np.array([line.split()[2] for line in lines]).astype(int)
        return nodes, labels


def adjacency_matrix(edges: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """
    Creates an adjacency matrix given an array of edges and array of nodes,
    where A_ij=1 if nodes i and j are connected, otherwise 0.

    This method is quick enough for most purposes.

    You could also utilize Scipy's sparse COO matrix:
        n = nodes.shape[0]
        i = edges[:, 0]-1
        j = edges[:, 1]-1
        v = np.ones(edges.shape[0])
        A = sparse.coo_matrix((v, (i, j)), shape=(n, n))
        A = (A + np.transpose(A))/2
        (The second line ensures the output graph representation is undirected.)

    :param edges: Array of graph edges.
    :param nodes: Array of graph nodes.
    :return: An adjacency matrix (array).
    """
    adj_matrix = np.zeros((nodes.shape[0], nodes.shape[0]))
    data = edges - 1
    for row, col in data:
        adj_matrix[row, col] = 1
        adj_matrix[col, row] = 1
    return adj_matrix


class SpectralCluster:
    """Class for basic spectral clustering"""
    def __init__(self, adjacency_matrix: np.ndarray, k: int = 2):
        self.adjacency_matrix = adjacency_matrix
        self.k = k
        self.degree_matrix: Optional[np.ndarray] = None
        self.laplacian_matrix: Optional[np.ndarray] = None
        self.eig_vals: Optional[np.ndarray] = None
        self.eig_vecs: Optional[np.ndarray] = None
        self.features: Optional[np.ndarray] = None
        self.model: Optional[KMeans] = None

    @classmethod
    def from_nodes_edges(cls, nodes: np.ndarray, edges: np.ndarray,
                         k: int = 2) -> "SpectralCluster":
        """Allows construction directly from nodes and edges"""
        adj_matrix = np.zeros((nodes.shape[0], nodes.shape[0]))
        data = edges - 1
        for row, col in data:
            adj_matrix[row, col] = 1
            adj_matrix[col, row] = 1
        return cls(adj_matrix, k)

    def get_degree_matrix(self) -> np.ndarray:
        """Returns degree matrix"""
        deg_mat = np.diag(self.adjacency_matrix.sum(axis=1))
        setattr(self, 'degree_matrix', deg_mat)
        return deg_mat

    def get_laplacian_matrix(self) -> np.ndarray:
        """Returns graph Laplacian"""
        laplacian = self.get_degree_matrix() - self.adjacency_matrix
        setattr(self, 'laplacian_matrix', laplacian)
        return laplacian

    def compute_eigenvectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns eigenvalues and eigenvectors"""
        eig_vals, eig_vecs = scipy.linalg.eig(self.get_laplacian_matrix())
        setattr(self, 'eig_vals', eig_vals.real)
        setattr(self, 'eig_vecs', eig_vecs.real)
        return eig_vals.real, eig_vecs.real

    def get_eigvals_eigvecs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns sorted eigenvalues and eigenvectors"""
        eig_vals, eig_vecs = self.compute_eigenvectors()
        sort_idx = np.argsort(eig_vals)
        sorted_vecs = eig_vecs[:, sort_idx]
        sorted_vals = eig_vals[sort_idx]
        return sorted_vals, sorted_vecs

    def get_features(self, k_eig_vecs: int):
        """Selects K eigenvectors to use as features in K-Means"""
        eig_vals, eig_vecs = self.get_eigvals_eigvecs()
        features = eig_vecs[:, :k_eig_vecs]
        setattr(self, 'features', features)
        return features

    def fit(self, k_eig_vecs: Optional[int] = None):
        """Instantiates and fits a K-Means object"""
        kmeans = KMeans(n_clusters=self.k)
        kmeans.fit(self.get_features(k_eig_vecs))
        setattr(self, 'model', kmeans)
        return kmeans
