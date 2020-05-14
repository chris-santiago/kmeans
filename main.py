"""Main module"""
import numpy as np

from k_means.k_means_cluster import KMeansCluster
from k_means.k_means_numpy import KMeans
from k_means.k_medoids_cluster import KMedoidsCluster
from k_means.make_clusters import SAMPLE_DATA


def run_kmeans():
    """Implement K-Means using generated data"""
    kmeans = KMeansCluster(k=3, max_iter=100, method='manhattan')
    kmeans.fit(SAMPLE_DATA, verbose=1)
    kmeans.plot()


def run_kmediods():
    kmeans = KMedoidsCluster(k=3, max_iter=100, method='euclidean')
    kmeans.fit(SAMPLE_DATA, verbose=2)
    kmeans.plot()


def from_text(file):
    points = []
    file = 's2.txt'
    with open(file, 'r') as text_file:
        for line in text_file.readlines():
            points.append(tuple(line.strip().split('    ')))
    points = np.array([(int(a), int(b)) for a, b in points])
    kmeans = KMeans(k=15)
    kmeans.fit(points, verbose=False)
    kmeans.plot()


if __name__ == '__main__':
    from_text('s2.txt')
