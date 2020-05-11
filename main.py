"""Main module"""
from k_means.k_means_cluster import KMeansCluster
from k_means.make_clusters import SAMPLE_DATA


def main():
    """Implement K-Means using generated data"""
    kmeans = KMeansCluster(k=3, max_iter=100)
    kmeans.fit(SAMPLE_DATA, verbose=True)
    kmeans.plot()


main()
