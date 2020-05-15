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
import warnings
from typing import Tuple, Optional

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode

from k_means_numpy import KMeans
from make_clusters import SAMPLE_DATA

warnings.filterwarnings('ignore', category=da.core.PerformanceWarning)


class KMedoids(KMeans):
    """A vectorized K-Medoids Clustering Model using Numpy"""
    def __init__(self, k: int = 2, tol: float = 0.001, max_iter: int = 300,
                 method: str = 'euclidean'):
        super().__init__(k, tol, max_iter, method)
        self.batch_runs: Optional[np.ndarray] = None

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

    def dask_update_centroids(self, data: np.ndarray,
                              chunk_size: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
        """Modifies `update_centroids(_)` function to use Dask arrays"""
        old_centroids = self.centroids.copy()
        dask_array = da.from_array(data, chunks=chunk_size)
        for cluster in range(self.k):
            in_cluster = np.where(self.clusters == cluster)
            distance_matrix = self.get_distance_vec(dask_array[in_cluster], dask_array[in_cluster])
            min_wcss = np.argmin(distance_matrix.sum(axis=0))
            self.centroids[cluster, :] = dask_array[in_cluster][min_wcss]
        new_centroids = self.centroids.copy()
        return old_centroids, new_centroids

    @staticmethod
    def _batch_data(data: np.ndarray, batch_size: int) -> np.ndarray:
        """Method to batch data by random sampling"""
        indices = np.random.randint(data.shape[0], size=batch_size)
        return data[indices]

    @staticmethod
    def get_n_batches(data, batch_size):
        """Determine a default number of batches"""
        return int((data.shape[0] / batch_size) * 1.25)

    def initialize_batches(self, n_batches: int, data: np.ndarray) -> None:
        """Method to initialize array for batch runs"""
        self.batch_runs = np.zeros((n_batches, self.k, data.shape[1]))

    def set_final_centroids(self):
        """Set final centroids for batch runs using mode as measure of centrality"""
        # subscript mode with [0] to return 2D matrix
        setattr(self, 'centroids', mode(self.batch_runs).mode[0])

    def set_final_assignments(self, data):
        """Set final cluster assignments for batch runs using final centroids"""
        self.assign_cluster_vec(data)
        assignments = np.append(self.clusters.reshape(-1, 1), data, axis=1)
        setattr(self, 'assignments', assignments)

    def fit(self, data: np.ndarray, verbose: int = 1) -> "KMedoids":
        """
        ** Uses Dask Arrays **
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
            old_centroids, new_centroids = self.dask_update_centroids(data)
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

    def fit_batch(self, data: np.ndarray, verbose: int = 1,
                  batch_size: int = 6400, n_batches: Optional[int] = None) -> "KMeans":
        """
        Function to fit K-Means object to dataset using mini-batches.
        Randomly chooses initial centroids, assigns datapoints.
        Updates centroids and re-assigns datapoints.
        Continues until algorithm converges and WCSS is minimized.

        :param data: Numpy array of data
        :param verbose: Integer indicating verbosity of printouts
        :param batch_size:
        :param n_batches:
        :return: self
        """
        if verbose not in {0, 1, 2}:
            raise ValueError('Verbose must be set to {0, 1, 2}')
        if n_batches is None:
            n_batches = self.get_n_batches(data, batch_size)
        self.intialize_centroids(data)
        self.initialize_batches(n_batches, data)
        print(f'Running {n_batches} batches of size {batch_size}...')
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
                    print(f'Iteration: {i}, '
                          f'WCSS: {self.within_cluster(self.centroids, batch_data)}')
                i += 1
            else:
                raise RuntimeError('Failed to converge!')
            self.batch_runs[n, :, :] = self.centroids
        self.set_final_centroids()
        self.set_final_assignments(data)
        return self

    def plot(self) -> None:
        """Plot clusters and circles medoid in red"""
        for cluster in range(self.k):
            cluster_filter = self.assignments[:, 0] == cluster
            plt.scatter(self.assignments[cluster_filter, 1],
                        self.assignments[cluster_filter, 2],
                        alpha=0.5)
        for point in self.centroids:
            plt.scatter(point[0], point[1], marker='o', edgecolors='black', facecolors='none')
        plt.title(f'Clustering for {self.k} Medoids (scaled)')
        plt.show()


def main():
    """Main function"""
    kmedoids = KMedoids(k=5)
    kmedoids.fit_batch(SAMPLE_DATA)
    # kmedoids.fit_batch(SAMPLE_DATA, n_batches=15, batch_size=5000)
    print('Final medoids:')
    print(kmedoids.centroids)
    kmedoids.plot()


if __name__ == '__main__':
    main()
