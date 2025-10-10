# (c) 2023, CEA LIST
#
# All rights reserved.
# SPDX-License-Identifier: MIT
#
# Contributors
#    Based on https://github.com/fregu856/deeplabv3
#    Fabio Arnez, probabilistic adaptation
#    Daniel Montoya

from torch import Tensor
from entropy_estimators import continuous
import numpy as np
from typing import Tuple, Union
from tqdm.contrib.concurrent import process_map

__all__ = ["get_dl_h_z", "single_image_entropy_calculation"]


def single_image_entropy_calculation(sample: np.ndarray, neighbors: int) -> np.ndarray:
    """
    Function used to calculate the entropy values of a single image. Used to calculate entropy
    in parallel

    Args:
        sample: MCD samples for a single image
        neighbors: Number of neighbors to perform calculation. By default 5 seems ok, but should
            be n-1 if n<=5

    Returns:
        Entropy of the activations for a single image
    """
    h_z_batch = []
    for z_val_i in range(sample.shape[1]):
        h_z_i = continuous.get_h(sample[:, z_val_i], k=neighbors, norm="max", min_dist=1e-5)
        h_z_batch.append(h_z_i)
    h_z_batch_np = np.asarray(h_z_batch)
    return h_z_batch_np


def get_dl_h_z(
    dl_z_samples: Union[Tensor, np.ndarray], mcd_samples_nro: int = 32, parallel_run: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get dataloader Entropy $h(.)$ for Z, from Monte Carlo Dropout (MCD) samples

    Args:
        dl_z_samples: Dataloader Z Samples
        mcd_samples_nro: Number of monte carlo dropout samples
        parallel_run: Optionally try to perform entropy calculations in parallel

    Returns:
        Latent vector multivariate normal entropy $h(Z)$, Latent vector value entropy $h(z_i)$
    """
    if not isinstance(dl_z_samples, Tensor):
        assert isinstance(dl_z_samples, np.ndarray), (
            "dl_z_samples must be a torch Tensor" " or numpy array"
        )
    assert isinstance(mcd_samples_nro, int), "mcd_samples_nro must be an integer"
    assert isinstance(parallel_run, bool), "parallel_run must be a boolean"
    # Get dataloader mvn h(z), from mcd_samples
    if isinstance(dl_z_samples, Tensor):
        z_samples_ls = [i for i in dl_z_samples.split(mcd_samples_nro)]
        z_samples_np_ls = [t.cpu().numpy() for t in z_samples_ls]
    else:
        z_samples_ls = [
            i for i in np.split(dl_z_samples, int(dl_z_samples.shape[0] / mcd_samples_nro))
        ]
        z_samples_np_ls = [t for t in z_samples_ls]
    # Choose correctly the number of neighbors for the entropy calculations:
    # It has to be smaller than the mcd_samples_nro by at least 1
    k_neighbors = 5 if mcd_samples_nro > 5 else mcd_samples_nro - 1
    dl_h_mvn_z_samples_ls = [
        continuous.get_h(s, k=k_neighbors, norm="max", min_dist=1e-5) for s in z_samples_np_ls
    ]
    dl_h_mvn_z_samples_np = np.array(dl_h_mvn_z_samples_ls)
    dl_h_mvn_z_samples_np = np.expand_dims(dl_h_mvn_z_samples_np, axis=1)
    # Get dataloader entropy $h(z_i)$ for each value of Z, from mcd_samples
    if not parallel_run:
        dl_h_z_samples = []
        for input_mcd_samples in z_samples_np_ls:
            h_z_batch = []
            for z_val_i in range(input_mcd_samples.shape[1]):
                # h_z_i = continuous.get_h(input_mcd_samples[:, z_val_i], k=5)  # old
                h_z_i = continuous.get_h(
                    input_mcd_samples[:, z_val_i], k=k_neighbors, norm="max", min_dist=1e-5
                )
                h_z_batch.append(h_z_i)
            h_z_batch_np = np.asarray(h_z_batch)
            dl_h_z_samples.append(h_z_batch_np)
    else:
        dl_h_z_samples = process_map(
            single_image_entropy_calculation,
            z_samples_np_ls,
            [k_neighbors] * len(z_samples_np_ls),
            chunksize=1,
        )
    dl_h_z_samples_np = np.asarray(dl_h_z_samples)
    return dl_h_mvn_z_samples_np, dl_h_z_samples_np
