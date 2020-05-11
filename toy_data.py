"""Module for creating toy cluster data"""
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')


TOY_DATA = np.array([
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [1, 0.6],
    [9, 11]
])


def main():
    """Main method for plotting data"""
    plt.scatter(TOY_DATA[:, 0], TOY_DATA[:, 1], s=150)
    plt.show()


if __name__ == '__main__':
    main()
