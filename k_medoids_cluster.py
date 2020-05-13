"""
A simple K-Medoids implementation

Methodology for the K Means algorithm:

    Choose value for K
    Randomly select K featuresets to start as your centroids
    Calculate distance of all other featuresets to centroids
    Classify other featuresets as same as closest centroid
    Determine point within centroid that minimizes within-cluster distance function, making that
    the new centroid
    Repeat steps 3-5 until optimized (centroids no longer moving)
"""
from typing import Dict, List, Tuple, Union

import numpy as np
from pandas import DataFrame

from k_means.make_clusters import SAMPLE_DATA
from k_means_cluster import KMeansCluster

ClusterData = Union[List, np.array, np.ndarray, DataFrame]
ClusterPoint = Tuple[Union[int, float], ...]


class KMedoidsCluster(KMeansCluster):
    def __init__(self, k: int = 2, tol: float = 0.0001, max_iter: int = 300,
                 method: str = 'euclidean'):
        super().__init__(k, tol, max_iter, method)

    def get_distance(self, arr_1: ClusterData, arr_2: ClusterData) -> float:
        """Calculate distance between two points"""
        arr_1 = self._convert_to_array(arr_1)
        arr_2 = self._convert_to_array(arr_2)
        if self.method == 'manhattan':
            dist = np.linalg.norm(arr_1 - arr_2, ord=1)
        else:
            dist = np.linalg.norm(arr_1 - arr_2)
        return dist

    def get_medoid_distances(self, cluster: int) -> Dict[ClusterPoint, float]:
        """
        Calculates, for each point in a cluster, the distance between itself and other points.

        :param cluster: Cluster to compute distances within
        :return: A dictionary of points and sum of distances to within-cluster points
        """
        medoid_distances = {}
        for point in self.clusters[cluster]:
            medoid_distances[tuple(point)] = self.get_distance(point, self.clusters[cluster])
        return medoid_distances

    def update_centroid(self) -> "KMedoidsCluster":
        """Updates centroids by choosing point with smallest within-cluster distance"""
        for cluster in self.clusters:
            mediod_distances = self.get_medoid_distances(cluster)
            new_medoid = [np.array(point) for point, distance in mediod_distances.items()
                          if distance == min(mediod_distances.values())]
            self.centroids[cluster] = new_medoid[0]
        return self


def main():
    """Main function"""
    kmediods = KMedoidsCluster(k=3, method='manhattan')
    kmediods.fit(SAMPLE_DATA, verbose=1)
    kmediods.plot()
    kmediods.eval_metrics()


if __name__ == '__main__':
    main()
