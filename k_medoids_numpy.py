"""
A Numpy K-Medoids implementation

Methodology for the K-Medoids algorithm:

    Choose value for K
    Randomly select K featuresets to start as your centroids
    Calculate distance of all other featuresets to centroids
    Classify other featuresets as same as closest centroid
    Determine point within centroid that minimizes within-cluster distance function, making that
    the new centroid
    Repeat steps 3-5 until optimized (centroids no longer moving)
"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode

from k_means_numpy import KMeans
from make_clusters import SAMPLE_DATA


class KMedoids(KMeans):
    """A vectorized K-Medoids Clustering Model using Numpy"""
    def __init__(self, k: int = 2, tol: float = 0.001, max_iter: int = 300,
                 method: str = 'euclidean'):
        super().__init__(k, tol, max_iter, method)
        self.batch_runs = None

    def update_centroids(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Function to update centroids (medoids) by computing a distance matrix for each point
        and other points within a given cluster.  Use `np.argmin()` and sum of distance matrix
        columns to find column-index with lowest WCSS.

        Returns a tuple of old centroids and new centroids for convergence detection and
        updates the `centroids` instance attribute.
        """
        old_centroids = self.centroids.copy()
        for cluster in range(self.k):
            in_cluster = np.where(self.clusters == cluster)
            distance_matrix = self.get_distance_vec(data[in_cluster], data[in_cluster])
            min_wcss = np.argmin(distance_matrix.sum(axis=0))
            self.centroids[cluster, :] = data[in_cluster][min_wcss]
        new_centroids = self.centroids.copy()
        return old_centroids, new_centroids

    def _batch_data(self, data, batch_size):
        indices = np.random.randint(data.shape[0], size=batch_size)
        return data[indices]

    def fit(self, data, verbose=1, n_batches=10, batch_size=6400) -> "KMeans":
        """
        Function to fit K-Means object to dataset using mini-batches.
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
        self.batch_runs = np.zeros((n_batches, self.k, data.shape[1]))
        for n in range(n_batches):
            batch_data = self._batch_data(data, batch_size)
            i = 1
            while i <= self.max_iter:
                self.assign_cluster_vec(batch_data)
                old_centroids, new_centroids = self.update_centroids(batch_data)
                if verbose > 1:
                    self.print_assignments(self.clusters, batch_data)
                if self.meets_tolerance(old_centroids, new_centroids):
                    self.wcss = self.within_cluster(self.centroids, batch_data)
                    self.assignments = np.append(self.clusters.reshape(-1, 1), batch_data, axis=1)
                    print(f'Converged in {i-1} iterations.  WCSS: {self.wcss}')
                    break
                if verbose > 0:
                    print(f'Iteration: {i}, WCSS: {self.within_cluster(self.centroids, batch_data)}')
                i += 1
            self.batch_runs[n, :, :] = self.centroids
        self.centroids = mode(self.batch_runs).mode
        return self

    def plot(self) -> None:
        """Plot clusters and circles medoid in red"""
        for cluster in range(self.k):
            cluster_filter = self.assignments[:, 0] == cluster
            plt.scatter(self.assignments[cluster_filter, 1],
                        self.assignments[cluster_filter, 2],
                        alpha=0.5)
        for point in self.centroids:
            plt.scatter(point[0], point[1], marker='o', edgecolors='r', facecolors='none')
        plt.show()


def main():
    """Main function"""
    kmedoids = KMedoids(k=3)
    kmedoids.fit(SAMPLE_DATA, n_batches=15, batch_size=5000)
    print('Final medoids:')
    print(kmedoids.centroids)
    print('Batch results:')
    print(kmedoids.batch_runs)


if __name__ == '__main__':
    main()
