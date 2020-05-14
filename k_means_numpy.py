"""
A Numpy K-Means implementation

Methodology for the K Means algorithm:

    Choose value for K
    Randomly select K featuresets to start as your centroids
    Calculate distance of all other featuresets to centroids
    Classify other featuresets as same as closest centroid
    Take mean of each class (mean of all featuresets by class), making that mean the new centroid
    Repeat steps 3-5 until optimized (centroids no longer moving)
"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from k_means.make_clusters import SAMPLE_DATA


class KMeans:
    """A vectorized K-Means Clustering Model using Numpy"""

    def __init__(self, k: int = 2, tol: float = 0.001, max_iter: int = 300,
                 method: str = 'euclidean'):
        if method not in {'euclidean', 'manhattan'}:
            raise ValueError('Method must be one of "euclidean" or "manhattan"')
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.method = method

        self.centroids: np.ndarray = np.array(0)
        self.clusters: np.ndarray = np.array(0)
        self.assignments: np.ndarray = np.array(0)
        self.wcss: float = 0.0

    def intialize_centroids(self, data: np.ndarray) -> "KMeans":
        """Get initial centroids"""
        centroids = data.copy()
        np.random.shuffle(data)
        self.centroids = centroids[:self.k]
        return self

    def get_distance(self, data: np.ndarray) -> np.ndarray:
        """Calculate distance between two points"""
        choose_method = {'euclidean': 2, 'manhattan': 1}
        dist = np.zeros((data.shape[0], self.centroids.shape[0]))
        for centroid in range(self.centroids.shape[0]):
            for point in range(data.shape[0]):
                dist[point, centroid] = np.linalg.norm(data[point, :] - self.centroids[centroid, :],
                                                       ord=choose_method[self.method])
        return dist

    def get_distance_vec(self, data: np.ndarray) -> np.ndarray:
        """Calculate distance between two arrays"""
        if self.method == 'manhattan':
            return np.abs(self.centroids - data[:, np.newaxis, :]).sum(axis=2)
        return np.sqrt(((self.centroids - data[:, np.newaxis, :]) ** 2).sum(axis=2))

    def assign_cluster(self, data: np.ndarray) -> "KMeans":
        """Function to assign points to a centroid"""
        self.clusters = np.argmin(self.get_distance(data), axis=1)
        return self

    def assign_cluster_vec(self, data: np.ndarray) -> "KMeans":
        """Function to assign points to a centroid"""
        self.clusters = np.argmin(self.get_distance_vec(data), axis=1)
        return self

    def update_centroids(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Function to update centroids"""
        old_centroids = self.centroids.copy()
        for cluster in range(self.k):
            self.centroids[cluster, :] = np.mean(data[self.clusters == cluster],
                                                 axis=0)
        new_centroids = self.centroids.copy()
        return old_centroids, new_centroids

    def within_cluster(self, data):
        """Calculates within cluster sum of distances"""
        return sum(np.amin(self.get_distance_vec(data), axis=1))

    def meets_tolerance(self, old_centroids, new_centroids):
        """Function to detect convergence"""
        return np.linalg.norm(old_centroids - new_centroids) <= self.tol

    def fit(self, data, verbose=True) -> "KMeans":
        """
        Function to fit K-Means object to dataset.
        Randomly chooses initial centroids, assigns datapoints.
        Updates centroids and re-assigns datapoints.
        Continues until algorithm converges and WCSS is minimized.

        :param data: Numpy array of data
        :param verbose: Boolean indicating verbosity of printouts
        :return: self
        """
        self.intialize_centroids(data)
        i = 1
        while i <= self.max_iter:
            self.assign_cluster_vec(data)
            old_centroids, new_centroids = self.update_centroids(data)
            if self.meets_tolerance(old_centroids, new_centroids):
                self.wcss = self.within_cluster(data)
                self.assignments = np.append(self.clusters.reshape(-1, 1), data, axis=1)
                print(f'Converged in {i-1} iterations.  WCSS: {self.wcss}')
                break
            if verbose:
                print(f'Iteration: {i}, WCSS: {self.within_cluster(data)}')
            i += 1
        return self

    def plot(self) -> None:
        """Plot clusters and centroids"""
        for cluster in range(self.k):
            cluster_filter = self.assignments[:, 0] == cluster
            plt.scatter(self.assignments[cluster_filter, 1],
                        self.assignments[cluster_filter, 2],
                        alpha=0.5)
        for point in self.centroids:
            plt.scatter(point[0], point[1], marker='x', s=100, c='red')
        plt.show()


def main():
    """Main function"""
    kmeans = KMeans(k=3)
    kmeans.fit(SAMPLE_DATA).plot()


if __name__ == '__main__':
    main()
