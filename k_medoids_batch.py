"""
A Batch K-Medoids implementation

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
from typing import Tuple, Optional, Dict

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode

from k_medoids_numpy import KMedoids
from make_clusters import SAMPLE_DATA

from sklearn.metrics import pairwise_distances
import scipy.spatial.distance

warnings.filterwarnings('ignore', category=da.core.PerformanceWarning)


class BatchKMedoids(KMedoids):
    """A batch K-Medoids Clustering Model using Numpy"""
    def __init__(self, k: int = 2, tol: float = 0.001, max_iter: int = 300,
                 method: str = 'euclidean'):
        super().__init__(k, tol, max_iter, method)
        self.batch_runs: Optional[np.ndarray] = None

    def get_distance(self, centroids: np.ndarray, data: np.ndarray) -> np.ndarray:
        """
        Overwrites base class method; the Scipy cdist doesn't work well with the batches--
        it seems to create medoids where points do not exist.

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
        if self.method == 'cityblock':
            return np.abs(centroids - data[:, np.newaxis, :]).sum(axis=2)
        return np.sqrt(((centroids - data[:, np.newaxis, :]) ** 2).sum(axis=2))

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
        self.assign_cluster(data)
        assignments = np.append(self.clusters.reshape(-1, 1), data, axis=1)
        setattr(self, 'assignments', assignments)

    def fit(self, data: np.ndarray, verbose: int = 1, batch_size: int = 6400,
            n_batches: Optional[int] = None) -> "BatchKMedoids":
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
                self.assign_cluster(batch_data)
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


def main():
    """Main function"""
    import time
    start = time.time()
    kmedoids = BatchKMedoids(k=5)
    kmedoids.fit(SAMPLE_DATA, verbose=0)
    print('Final medoids:')
    print(kmedoids.centroids)
    end = time.time()
    print(f'Time: {end - start}')
    kmedoids.plot()


if __name__ == '__main__':
    main()