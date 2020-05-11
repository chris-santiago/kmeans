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
from typing import Dict, Any, Union, List
from collections import defaultdict
import numpy as np
from pandas import DataFrame
from k_means.toy_data import TOY_DATA


ClusterData = Union[List, np.array, np.ndarray, DataFrame]


class KMeansCluster:
    """Simple K-Means Clustering Model"""
    def __init__(self, k: int = 2, tol: float = 0.001, max_iter: int = 300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

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
        return np.linalg.norm(arr_1 - arr_2)

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

    def meets_tolerance(self, old_centroids: np.ndarray, new_centroids: np.ndarray) -> bool:
        """Function to detect convergence"""
        return np.abs(self.get_distance(old_centroids, new_centroids)) <= self.tol

    def fit(self, data: ClusterData):
        """
        Function to fit K-Means object to dataset.

        :param data: Dataset of type [list, DataFrame or NDArray]
        :return: Dictionary of clusters and points
        """
        data = self._convert_to_array(data)
        self.initialize_centroids(data).assign_cluster(data)
        print(f'Initial clusters: {self.clusters.items()}')
        i = 0
        while i <= self.max_iter:
            old_centroids = list(self.centroids.values())
            new_centroids = list(self.update_centroid().centroids.values())
            if self.meets_tolerance(old_centroids, new_centroids):
                break
            self.assign_cluster(data)
            i += 1
        print(f'Final clusters: {self.clusters.items()}')
        return self.clusters.items()


def main():
    """Main function"""
    kmeans = KMeansCluster()
    kmeans.fit(TOY_DATA)


if __name__ == '__main__':
    main()
