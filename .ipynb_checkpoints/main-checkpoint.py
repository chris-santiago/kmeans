"""Main module for GATech grading purposes"""
from typing import Tuple

import numpy as np

from image import rescale_image, scale_image
from k_means_numpy import KMeans
from k_medoids_numpy import KMedoids
from make_clusters import SAMPLE_DATA


def test_kmeans(pixels: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper for GATech grading.

    :param pixels: A numpy array of image data
    :param k: Number of clusters to use
    :return: A tuple of classes and centroids
    """
    scaled_data = scale_image(pixels)
    kmeans = KMeans(k=k)
    kmeans.fit(scaled_data, verbose=0)
    return kmeans.clusters, rescale_image(kmeans.centroids)


def test_kmedoids(pixels: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper for GATech grading.

    :param pixels: A numpy array of image data
    :param k: Number of clusters to use
    :return: A tuple of classes and centroids
    """
    scaled_data = scale_image(pixels)
    kmedoids = KMedoids(k=k)
    kmedoids.fit(scaled_data, verbose=0)
    return kmedoids.clusters, rescale_image(kmedoids.centroids)


if __name__ == '__main__':
    classes, centroids = test_kmedoids(SAMPLE_DATA, 5)
    print(classes)
    print(centroids)
