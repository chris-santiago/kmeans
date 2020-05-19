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
import time
import warnings
from typing import Optional, Tuple

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform

from k_means_numpy import KMeans
from make_clusters import SAMPLE_DATA

warnings.filterwarnings('ignore', category=da.core.PerformanceWarning)


class KMedoids(KMeans):
    """A vectorized K-Medoids Clustering Model using Numpy"""
    # def __init__(self, k: int = 2, tol: float = 0.001, max_iter: int = 300,
    #              method: str = 'euclidean'):
    #     super().__init__(k, tol, max_iter, method)
    #     self.batch_runs: Optional[np.ndarray] = None

    def soft_initialization(self, data: np.ndarray, sample_size: int = 3000) -> "KMedoids":
        """
        Initialize medoids using a random sample of points.
        Runs an initial k-mediods iteration on a random subset of data.
        """
        subset_idx = np.random.randint(data.shape[0], size=sample_size)
        subset = data[subset_idx]
        centroid_idx = np.random.randint(subset.shape[0], size=self.k)
        self.centroids = subset.copy()[centroid_idx]
        i = 1
        while i <= self.max_iter:
            self.assign_cluster(subset)
            old_centroids, new_centroids = self.update_centroids(subset)
            if self.meets_tolerance(old_centroids, new_centroids):
                self.wcss = self.within_cluster(self.centroids, subset)
                self.assignments = np.append(self.clusters.reshape(-1, 1), subset, axis=1)
                break
            i += 1
        return self

    def manual_initialization(self, points: np.ndarray) -> "KMedoids":
        """Method to allow manually set centroids"""
        self.centroids = points
        return self

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
            # assuming scipy distance calcs are optimized...
            # distance_matrix = self.get_distance_vec(data[in_cluster], data[in_cluster])
            distance_matrix = squareform(pdist(data[in_cluster], self.method))
            min_wcss = np.argmin(distance_matrix.sum(axis=0))
            self.centroids[cluster, :] = data[in_cluster][min_wcss]
        new_centroids = self.centroids.copy()
        return old_centroids, new_centroids

    def find_nearest_neighbors(self, centroid: np.ndarray, cluster_points: np.ndarray,
                               n_neighbors: int) -> np.ndarray:
        """Finds k nearest points to current centroid"""
        distances = self.get_distance(centroid, cluster_points)
        sorted_indices = np.argsort(distances.flatten(), axis=0)
        return cluster_points[sorted_indices][:n_neighbors]

    def nn_update_centroids(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Modifies `update_centroids()` function to search subset of nearest points to centroid"""
        old_centroids = self.centroids.copy()
        for cluster in range(self.k):
            in_cluster = np.where(self.clusters == cluster)
            nearest_neighbors = self.find_nearest_neighbors(self.centroids[cluster],
                                                            data[in_cluster], 1000)
            distance_matrix = self.get_distance(nearest_neighbors, data[in_cluster])
            min_wcss = np.argmin(distance_matrix.sum(axis=0))
            self.centroids[cluster, :] = nearest_neighbors[min_wcss]
        new_centroids = self.centroids.copy()
        return old_centroids, new_centroids

    def _get_distances_chunked(self, data: np.ndarray, chunk_size: int) -> np.ndarray:
        """Calculates pairwise distances in chunks"""
        res = np.zeros((1, 2))
        n_rows = data.shape[0]
        for i in range(0, n_rows, chunk_size):
            chunk = data[i:i + chunk_size, :]
            chunk_dist = self.get_distance(chunk, data)
            min_idx = np.argmin(chunk_dist.sum(axis=0))
            chunk_sum = chunk_dist[:, min_idx].sum()
            chunk_idx = i + min_idx
            chunk_res = np.array([chunk_idx, chunk_sum])
            res = np.vstack((res, chunk_res))
        return res[1:, :]

    @staticmethod
    def _get_min_wcss_index(distances: np.ndarray) -> int:
        """Gets key with minimum value"""
        keys = distances[:, 0]
        vals = distances[:, 1]
        i = np.argsort(vals, axis=0)
        return int(keys[i][0])

    def batch_update_centroids(self, data: np.ndarray,
                               chunk_size: int = 6400) -> Tuple[np.ndarray, np.ndarray]:
        """Modifies `update_centroids()` function to use batched pairwise distances"""
        old_centroids = self.centroids.copy()
        for cluster in range(self.k):
            in_cluster = np.where(self.clusters == cluster)
            distances = self._get_distances_chunked(data[in_cluster], chunk_size)
            min_wcss = self._get_min_wcss_index(distances)
            self.centroids[cluster, :] = data[in_cluster][min_wcss]
        new_centroids = self.centroids.copy()
        return old_centroids, new_centroids

    def dask_update_centroids(self, data: np.ndarray,
                              chunk_size: int = 6400) -> Tuple[np.ndarray, np.ndarray]:
        """Modifies `update_centroids(_)` function to use Dask arrays"""
        old_centroids = self.centroids.copy()
        dask_array = da.from_array(data, chunks=chunk_size)
        for cluster in range(self.k):
            in_cluster = np.where(self.clusters == cluster)
            # assuming sklearn/scipy distance calcs are optimized...
            distance_matrix = squareform(pdist(dask_array[in_cluster], self.method))
            min_wcss = np.argmin(distance_matrix.sum(axis=0))
            self.centroids[cluster, :] = dask_array[in_cluster][min_wcss]
        new_centroids = self.centroids.copy()
        return old_centroids, new_centroids

    def check_final_centroids(self, data):
        """Check that final centroids are existing data points"""
        for centroid in self.centroids:
            _is_in = (centroid == data).any()
            if _is_in is False:
                raise ValueError('Medoid cannot be assigned to non-existent point.')

    def _choose_initialization(self, soft_initialize: bool, data: np.ndarray,
                               initial_points: Optional[np.ndarray]) -> callable:
        """Method to choose initialization function"""
        methods = {
            True: self.soft_initialization,
            False: self.initialize_centroids,
        }
        if initial_points is None:
            return methods[soft_initialize](data)
        return self.manual_initialization(initial_points)

    def _choose_update_method(self, choice: str, data: np.ndarray) -> callable:
        """Method to choose centroid update function"""
        methods = {
            'brute': self.update_centroids,
            'batch': self.batch_update_centroids,
            'knn': self.nn_update_centroids,
            'dask': self.batch_update_centroids
        }
        return methods[choice](data)

    def fit(self, data: np.ndarray, verbose: int = 1, update_method: str = 'knn',
            soft_initialize: bool = True, initial_points: np.ndarray = None) -> "KMedoids":
        """
        ** Uses Dask Arrays **
        Function to fit K-Means object to dataset.
        Randomly chooses initial centroids, assigns datapoints.
        Updates centroids and re-assigns datapoints.
        Continues until algorithm converges and WCSS is minimized.

        :param initial_points:
        :param update_method:
        :param data: Numpy array of data
        :param verbose: Integer indicating verbosity of printouts
        :param use_dask: Boolean to use Dask arrays for pairwise distance calcs
        :param soft_initialize: Boolean to initialize centroids on random subset of data
        :return: self
        """
        if verbose not in {0, 1, 2}:
            raise ValueError('Verbose must be set to {0, 1, 2}')
        if update_method not in {'brute', 'batch', 'knn', 'dask'}:
            raise ValueError("Accepted update methods: {None, 'batch', 'knn', 'dask'}")
        self._choose_initialization(soft_initialize, data, initial_points)
        i = 1
        while i <= self.max_iter:
            self.assign_cluster(data)
            old_centroids, new_centroids = self._choose_update_method(update_method, data)
            if verbose > 1:
                self.print_assignments(self.clusters, data)
            if self.meets_tolerance(old_centroids, new_centroids):
                self.wcss = self.within_cluster(self.centroids, data)
                self.assignments = np.append(self.clusters.reshape(-1, 1), data, axis=1)
                print(f'Converged in {i-1} iterations.  WCSS: {self.wcss}')
                self.check_final_centroids(data)
                break
            if verbose > 0:
                print(f'Iteration: {i}, WCSS: {self.within_cluster(self.centroids, data)}')
            i += 1
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
    start = time.time()
    kmedoids = KMedoids(k=5, method='euclidean')
    kmedoids.fit(SAMPLE_DATA)
    print('Final medoids:')
    print(kmedoids.centroids)
    end = time.time()
    print(f'Time: {end - start}')
    kmedoids.plot()


if __name__ == '__main__':
    main()
