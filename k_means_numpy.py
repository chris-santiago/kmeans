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
from typing import Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance

from make_clusters import SAMPLE_DATA


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

        self.distance_mapper: Dict[str, str] = {
            'euclidean': 'euclidean',
            'manhattan': 'cityblock'
        }

    def intialize_centroids(self, data: np.ndarray) -> "KMeans":
        """Get initial centroids by random shuffle."""
        centroids = data.copy()
        indices = np.random.randint(data.shape[0], size=self.k)
        # np.random.shuffle(centroids)
        # self.centroids = centroids[:self.k]
        self.centroids = centroids[indices]
        return self

    def get_distance(self, centroids: np.ndarray, data: np.ndarray) -> np.ndarray:
        """
        Calculate a distance matrix, `S`, for two arrays so that  `s_ij` = d_ij` represents
        either the Euclidean or Manhattan distance from point `x_i` to centroid_j.

        The result is a matrix with shape [n_obs, n_clusters].

        This implementation first creates an empty matrix of shape [n_obs, n_clusters] and then
        loops through each centroid and point, assigning the distance for each point to one of the
        cluster columns.
        """
        choose_method = {'euclidean': 2, 'manhattan': 1}
        dist = np.zeros((data.shape[0], centroids.shape[0]))
        for centroid in range(centroids.shape[0]):
            for point in range(data.shape[0]):
                dist[point, centroid] = np.linalg.norm(data[point, :] - centroids[centroid, :],
                                                       ord=choose_method[self.method])
        return dist

    def get_distance_vec(self, centroids: np.ndarray, data: np.ndarray) -> np.ndarray:
        """
        Calculate a distance matrix, `S`, for two arrays so that  `s_ij` = d_ij` represents
        either the Euclidean or Manhattan distance from point `x_i` to centroid_j.

        The result is a matrix with shape [n_obs, n_clusters].

        This implementation adds a new axis in the data array, allowing for Numpy broadcasting
        with arrays of non-matching sizes.
            (If the two arrays differ in their number of dimensions, the shape of the array with
            fewer dimensions is padded with ones on its leading (left) side.)

        This broadcasting trick keeps the first dimension as the point or cluster dimension,
        allowing for for mathematical operations that reduce the number of matrix dims.
        """
        # if self.method == 'manhattan':
        #     return np.abs(centroids - data[:, np.newaxis, :]).sum(axis=2)
        # return np.sqrt(((centroids - data[:, np.newaxis, :]) ** 2).sum(axis=2))
        return scipy.spatial.distance.cdist(data, centroids, self.distance_mapper[self.method])

    def assign_cluster(self, data: np.ndarray) -> "KMeans":
        """
        Function to assign points to a centroid using a distance matrix as described in
        the `get_distance()` method.  Updates the `centroids` instance attribute.

        The `np.argmin()` function will select minimum distance across columns, assigning it to
        cluster=column-index.
        """
        self.clusters = np.argmin(self.get_distance(self.centroids, data), axis=1)
        return self

    def assign_cluster_vec(self, data: np.ndarray) -> "KMeans":
        """
        Function to assign points to a centroid using a distance matrix as described in
        the `get_distance_vec()` method.  Updates the `centroids` instance attribute.

        The `np.argmin()` function will select minimum distance across columns, assigning it to
        cluster=column-index.
        """
        self.clusters = np.argmin(self.get_distance_vec(self.centroids, data), axis=1)
        return self

    def update_centroids(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Function to update centroids by computing the mean of each cluster.
        Returns a tuple of old centroids and new centroids for convergence detection and
        updates the `centroids` instance attribute.
        """
        old_centroids = self.centroids.copy()
        for cluster in range(self.k):
            self.centroids[cluster, :] = np.mean(data[self.clusters == cluster],
                                                 axis=0)
        new_centroids = self.centroids.copy()
        return old_centroids, new_centroids

    def within_cluster(self, centroids: np.ndarray, data: np.ndarray) -> float:
        """
        Calculates within cluster sum of distances using a distance matrix as described in
        the `get_distance_vec()` method.

        This implementation uses `np.amin()` to find the minimum value in each column (cluster)
        and sum down the matrix.

        For example, suppose S is defined as follows:

            S = np.array([[0.3, 0.2],
                          [0.1, 0.5],
                          [0.4, 0.2]])

        Then WCSS(S) == 0.2 + 0.1 + 0.2 == 0.5.
        """
        return sum(np.amin(self.get_distance_vec(centroids, data), axis=1))

    def meets_tolerance(self, old_centroids: np.ndarray, new_centroids: np.ndarray) -> bool:
        """Function to detect convergence"""
        return np.linalg.norm(old_centroids - new_centroids) <= self.tol

    @staticmethod
    def print_assignments(clusters: np.ndarray, data: np.ndarray):
        """Print clusters and assigned points"""
        for point in np.append(clusters.reshape(-1, 1), data, axis=1):
            print(f'Cluster: {int(point[0])}, Point: {tuple(point[1:])}')

    def fit(self, data: np.ndarray, verbose: int = 1) -> "KMeans":
        """
        Function to fit K-Means object to dataset.
        Randomly chooses initial centroids, assigns datapoints.
        Updates centroids and re-assigns datapoints.
        Continues until algorithm converges and WCSS is minimized.

        :param data: Numpy array of data
        :param verbose: Integer indicating verbosity of printouts
        :return: self
        """
        if verbose not in {0, 1, 2}:
            raise ValueError('Verbose must be set to {0, 1, 2}')
        self.intialize_centroids(data)
        i = 1
        while i <= self.max_iter:
            self.assign_cluster_vec(data)
            old_centroids, new_centroids = self.update_centroids(data)
            if verbose > 1:
                self.print_assignments(self.clusters, data)
            if self.meets_tolerance(old_centroids, new_centroids):
                self.wcss = self.within_cluster(self.centroids, data)
                self.assignments = np.append(self.clusters.reshape(-1, 1), data, axis=1)
                print(f'Converged in {i-1} iterations.  WCSS: {self.wcss}')
                break
            if verbose > 0:
                print(f'Iteration: {i}, WCSS: {self.within_cluster(self.centroids, data)}')
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
        plt.title(f'Clustering for {self.k} Means (scaled)')
        plt.show()


def main():
    """Main function"""
    kmeans = KMeans(k=3)
    kmeans.fit(SAMPLE_DATA, verbose=0).plot()


if __name__ == '__main__':
    main()
