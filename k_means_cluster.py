"""
A simple K-Means implementation

Methodology for the K Means algorithm:

    Choose value for K
    Randomly select K featuresets to start as your centroids
    Calculate distance of all other featuresets to centroids
    Classify other featuresets as same as closest centroid
    Take mean of each class (mean of all featuresets by class), making that mean the new centroid
    Repeat steps 3-5 until optimized (centroids no longer moving)

"""
from collections import defaultdict
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from scipy.spatial.distance import cityblock

from k_means.toy_data import TOY_DATA

ClusterData = Union[List, np.array, np.ndarray, DataFrame]


class KMeansCluster:
    """Simple K-Means Clustering Model"""
    def __init__(self, k: int = 2, tol: float = 0.001, max_iter: int = 300,
                 method: str = 'euclidean'):
        if method not in {'euclidean', 'manhattan'}:
            raise ValueError('Method must be one of "euclidean" or "manhattan"')
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.method = method

        self.centroids: Dict[int, Any] = {}
        self.clusters: Dict[int, Any] = defaultdict(list)

    def initialize_centroids(self, data: ClusterData) -> "KMeansCluster":
        """Get initial centroids"""
        choices = np.random.choice([*range(len(data))], size=self.k, replace=False)
        for i in range(self.k):
            self.centroids[i] = data[choices[i]]
        return self

    @staticmethod
    def _convert_to_array(data: ClusterData) -> np.array:
        """Function to convert data into ndarray"""
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, List):
            return np.array(data)
        if isinstance(data, DataFrame):
            return DataFrame.values
        raise TypeError

    def get_distance(self, arr_1: ClusterData, arr_2: ClusterData):
        """Calculate distance between two points"""
        arr_1 = self._convert_to_array(arr_1)
        arr_2 = self._convert_to_array(arr_2)
        if self.method == 'manhattan':
            dist = cityblock(arr_1, arr_2)
        else:
            dist = np.linalg.norm(arr_1 - arr_2)
        return dist ** 2

    @staticmethod
    def find_min_idx(distances: List) -> int:
        """Find the index of the minimum distance"""
        return distances.index(min(distances))

    def assign_cluster(self, data: ClusterData) -> "KMeansCluster":
        """Function to assign a point to a centroid"""
        data = self._convert_to_array(data)
        self.clusters.clear()
        for point in data:
            distance = []
            for centroid in self.centroids.values():
                distance.append(self.get_distance(point, centroid))
            cluster = self.find_min_idx(distance)
            self.clusters[cluster].append(point)
        return self

    def update_centroid(self) -> "KMeansCluster":
        """Function to update centroids"""
        for cluster in self.clusters:
            new_centroid = np.mean(np.array(self.clusters[cluster]), axis=0)
            self.centroids[cluster] = new_centroid
        return self

    def meets_tolerance(self, old_centroids: ClusterData, new_centroids: ClusterData) -> bool:
        """Function to detect convergence"""
        return np.abs(self.get_distance(old_centroids, new_centroids)) <= self.tol

    def fit(self, data: ClusterData, verbose=0):
        """
        Function to fit K-Means object to dataset.

        :param data: Dataset of type [list, DataFrame or NDArray]
        :param verbose: Integer [0, 1, 2] indicating verbosity of printouts
        :return: Dictionary of clusters and points
        """
        if verbose not in {0, 1, 2}:
            raise ValueError('Verbose must be set to {0, 1, 2}')
        data = self._convert_to_array(data)
        self.initialize_centroids(data).assign_cluster(data)
        if verbose > 0:
            print(f'Initial clusters: {self.clusters.items()}')
        converged = {}
        i = 0
        while i <= self.max_iter:
            for centroid in self.centroids:
                old_centroid = self.centroids[centroid]
                new_centroid = self.update_centroid().centroids[centroid]
                converged[centroid] = self.meets_tolerance(old_centroid, new_centroid)
            if verbose > 1:
                print(converged.values())
            if all(converged.values()):
                break
            self.assign_cluster(data)
            i += 1
        if verbose > 0:
            print(f'Final clusters: {self.clusters.items()}')
            print(f'Converged in {i} iterations.')
        return self.clusters.items()

    def plot(self) -> None:
        """Plot clusters and centroids"""
        for cluster in self.clusters.values():
            plt.scatter(np.array(cluster)[:, 0], np.array(cluster)[:, 1], alpha=0.5)
        for point in self.centroids.values():
            plt.scatter(point[0], point[1], marker='^', s=100)
        plt.show()


def main():
    """Main function"""
    kmeans = KMeansCluster(method='manhattan')
    kmeans.fit(TOY_DATA)
    print(list(kmeans.clusters.values()))
    kmeans.plot()


if __name__ == '__main__':
    main()
