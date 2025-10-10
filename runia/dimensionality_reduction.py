# (c) 2023, CEA LIST
#
# All rights reserved.
# SPDX-License-Identifier: MIT
#
# Contributors
#    Fabio Arnez
#    Daniel Montoya

from typing import Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import pacmap
from sklearn.decomposition import PCA

__all__ = [
    "apply_pca_ds",
    "apply_pca_ds_split",
    "apply_pca_transform",
    "plot_samples_pacmap",
    "fit_pacmap",
    "apply_pacmap_transform",
]


def apply_pca_ds(
    train_samples: np.ndarray,
    test_samples: np.ndarray,
    nro_components: int = 16,
    svd_solver: str = "randomized",
    whiten: bool = True,
) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """
    Applies PCA dimensionality reduction to a dataset

    Args:
        train_samples: PCA train and fit samples
        test_samples: PCA test samples
        nro_components: PCA components number
        svd_solver: PCA SVD solver
        whiten: PCA whiten value

    Returns:
        Train and Test samples with reduced dimensionality
    """
    assert isinstance(train_samples, np.ndarray), "train_samples must be a numpy array"
    assert isinstance(test_samples, np.ndarray), "test_samples must be a numpy array"
    assert isinstance(nro_components, int)
    pca_dim_red = PCA(n_components=nro_components, svd_solver=svd_solver, whiten=whiten)
    train_ds = pca_dim_red.fit_transform(train_samples)
    test_ds = pca_dim_red.transform(test_samples)
    return train_ds, test_ds, pca_dim_red


def apply_pca_ds_split(
    samples: np.ndarray,
    nro_components: int = 16,
    svd_solver: str = "randomized",
    whiten: bool = True,
) -> Tuple[np.ndarray, PCA]:
    """
    Applies PCA dimensionality reduction to a dataset split

    Args:
        samples: Dataset split samples
        nro_components: PCA nnumber of components, defaults to 16
        svd_solver: PCA SVD solver, defaults to 'randomized'
        whiten: PCA whiten value, defaults to True

    Returns:
        Dataset samples with reduced dimensionality, PCA transformation object
    """
    assert isinstance(samples, np.ndarray), "samples must be a numpy array"
    assert isinstance(nro_components, int)
    pca_dim_red = PCA(n_components=nro_components, svd_solver=svd_solver, whiten=whiten)
    dataset_dim_red = pca_dim_red.fit_transform(samples)
    return dataset_dim_red, pca_dim_red


def apply_pca_transform(samples: np.ndarray, pca_transform: PCA) -> np.ndarray:
    """
    Transform new samples with an already trained PCA transformation

    Args:
        samples: New samples
        pca_transform: The trained PCA

    Returns:
        Transformed samples
    """
    assert isinstance(samples, np.ndarray), "samples must be a numpy array"
    assert isinstance(pca_transform, PCA), "pca_transform must be a PCA instance"
    samples_dim_red = pca_transform.transform(samples)
    return samples_dim_red


def plot_samples_pacmap(
    samples_ind: np.ndarray,
    samples_ood: np.ndarray,
    neighbors: int = 25,
    components: int = 2,
    title: str = "Plot Title",
    return_figure: bool = False,
) -> Union[None, plt.scatter]:
    """
    In-Distribution vs Out-of-Distribution Data Projection 2D Plot using PaCMAP algorithm.

    Args:
        samples_ind: In-Distribution (InD) samples numpy array
        samples_ood: Out-of-Distribution (OoD) samples numpy array
        neighbors: Number of nearest-neighbors considered for the PaCMAP algorithm
        components: Number of components in final reduction
        title: Plot tile
        return_figure: True if show picture to screen instead of returning it

    Returns:
        Either a plot or None (show the plot)
    """
    assert isinstance(samples_ind, np.ndarray), "samples_ind must be a numpy array"
    assert isinstance(samples_ood, np.ndarray), "samples_ood must be a numpy array"
    assert isinstance(neighbors, int)
    assert isinstance(components, int)
    assert isinstance(title, str)
    assert isinstance(return_figure, bool)

    samples_concat = np.concatenate((samples_ind, samples_ood))
    label_normal = np.zeros((samples_ind.shape[0], 1))
    label_anomaly = np.ones((samples_ood.shape[0], 1))
    labels = np.concatenate((label_normal, label_anomaly))
    embedding = pacmap.PaCMAP(
        n_components=components, n_neighbors=neighbors, MN_ratio=0.5, FP_ratio=2.0
    )
    samples_transformed = embedding.fit_transform(samples_concat, init="pca")

    # visualize the embedding
    fig, axes = plt.subplots()
    scatter = axes.scatter(
        samples_transformed[:, 0],
        samples_transformed[:, 1],
        cmap="brg",
        c=labels,
        s=1.5,
    )
    axes.set_title(title)
    axes.set_xlabel("PACMAP dimension 1")
    axes.set_ylabel("PACMAP dimension 2")
    axes.legend(
        handles=scatter.legend_elements()[0],
        labels=["In-Distribution", "Out-of-Distribution"],
    )
    if return_figure:
        return fig
    plt.show()


def fit_pacmap(
    samples_ind: np.array, neighbors: int = 25, components: int = 2
) -> Tuple[np.array, pacmap.PaCMAP]:
    """
    In-Distribution vs Out-of-Distribution Data Projection 2D Plot using PaCMAP algorithm.

    Args:
        samples_ind: In-Distribution (InD) samples numpy array
        neighbors: Number of components in the output
        components: Number of nearest-neighbors considered for the PaCMAP algorithm

    Returns:
        Transformed samples and embedding
    """
    assert isinstance(samples_ind, np.ndarray)
    assert isinstance(neighbors, int)
    assert isinstance(components, int)

    embedding = pacmap.PaCMAP(
        n_components=components, n_neighbors=neighbors, MN_ratio=0.5, FP_ratio=2.0
    )
    samples_transformed = embedding.fit_transform(samples_ind, init="pca")
    return samples_transformed, embedding


def apply_pacmap_transform(
    new_samples: np.array, original_samples: np.array, pm_instance: pacmap.PaCMAP
) -> np.array:
    """
    Use the already trained PaCMAP to transform new samples

    Args:
        new_samples: New samples to be transformed
        original_samples: The original samples used to train the algorithm
        pm_instance: Instance of the trained PaCMAP

    Returns:
        Transformed new samples
    """
    assert isinstance(new_samples, np.ndarray)
    assert isinstance(original_samples, np.ndarray)
    assert isinstance(pm_instance, pacmap.PaCMAP)
    return pm_instance.transform(X=new_samples, basis=original_samples)
