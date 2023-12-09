import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def draw_vector(v0, v1, ax=None):
    """

    Parameters
    ----------
    v0
    v1
    ax

    Returns
    -------

    """
    ax = ax or plt.gca()
    arrow_props=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0,
                     shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrow_props)

def plot_pca_arrows(pca: PCA,*, axis_equal: bool=True) -> None:
    """

    Parameters
    ----------
    pca: The PCA object
    axis_equal: Flag to set the axis equal or not

    Returns
    -------

    """
    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = vector * 3 * np.sqrt(length)
        draw_vector(pca.mean_, pca.mean_ + v)

    if axis_equal:
        plt.axis('equal')
    plt.show()