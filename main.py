"""Main module"""
from k_means.k_means_cluster import KMeansCluster
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
    with open(file, 'r') as text_file:
        for line in text_file.readlines():
            points.append(tuple(line.strip().split('    ')))
    points = [(int(a), int(b)) for a, b in points]
    kmeans = KMeansCluster(k=15, method='euclidean', tol=0.00001)
    kmeans.fit(points, verbose=1)
    kmeans.plot()


# from_text('s2.txt')
run_kmediods()
