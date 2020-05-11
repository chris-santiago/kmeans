"""Module for generating cluster datasets"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

SAMPLE_DATA, SAMPLE_CLUSTERS = make_blobs(n_samples=100, n_features=2, centers=3)


def main():
    """Main method for plotting data"""
    plt.scatter(SAMPLE_DATA[:, 0], SAMPLE_DATA[:, 1], s=150)
    plt.show()


if __name__ == '__main__':
    main()
