# (c) 2023, CEA LIST
#
# All rights reserved.
# SPDX-License-Identifier: MIT
#
# Contributors
#    Based on https://github.com/fregu856/deeplabv3
#    Fabio Arnez, probabilistic adaptation
#    Daniel Montoya
import warnings

import faiss
import torch
from omegaconf import DictConfig
from sklearn.covariance import EmpiricalCovariance
from scipy.special import logsumexp, softmax
from torch import Tensor
from typing import Tuple, Dict, List, Union
import numpy as np
from runia.baselines.from_model_inference import (
    RouteDICE,
    normalizer,
)

__all__ = [
    "remove_latent_features",
    "get_baselines_thresholds",
    "calculate_all_baselines",
    "generalized_entropy",
    "get_labels_from_logits",
    "gmm_fit",
]


def get_dice_score_from_features(
    fc_params: Dict[str, np.ndarray],
    ind_data_dict: Dict[str, np.ndarray],
    ood_data_dict: Dict[str, np.ndarray],
    ood_names: List[str],
    ood_baselines_dict: Dict[str, np.ndarray],
    percentile: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute DICE scores from pre-extracted features using a RouteDICE layer.

    The function instantiates a RouteDICE layer using the provided fully-connected
    parameters and computes energy scores (via log-sum-exp) for the In-Distribution
    (InD) validation set and for each Out-of-Distribution (OoD) dataset.

    Args:
        fc_params: Dictionary containing the final linear layer parameters with keys
            "weight" and "bias".
        ind_data_dict: Dictionary with InD data arrays. Expected keys include
            "train features", "valid features", and "train logits".
        ood_data_dict: Dictionary with OoD data arrays. Each OoD dataset should
            provide a "{ood_name} features" entry.
        ood_names: List of OoD dataset names present in `ood_data_dict`.
        ood_baselines_dict: Dictionary that will be populated with computed OoD
            baseline scores (keyed by "{ood_name} dice").
        percentile: Percentile parameter passed to RouteDICE to define routing.

    Returns:
        Tuple of (updated ind_data_dict, updated ood_baselines_dict) where the
        InD dictionary receives a "dice" entry and the OoD baselines dictionary
        receives entries for each OoD dataset under the key "{ood_name} dice".
    """

    print("Calculating DICE score")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(fc_params["weight"], np.ndarray) or isinstance(fc_params["bias"], np.ndarray):
        params_tensor = {"weight": Tensor(fc_params["weight"]), "bias": Tensor(fc_params["bias"])}
    else:
        params_tensor = {"weight": fc_params["weight"], "bias": fc_params["bias"]}

    # Get mean per feature dimension
    dice_info = Tensor(ind_data_dict["train features"]).mean(0).cpu().numpy()
    dice_layer = RouteDICE(
        in_features=ind_data_dict["train features"].shape[1],
        out_features=ind_data_dict["train logits"].shape[1],
        bias=True,
        p=percentile,
        info=dice_info,
    )
    dice_layer.load_state_dict(params_tensor)
    dice_layer.to(device)
    dice_layer.eval()

    # Valid set scores
    with torch.no_grad():
        ind_valid_logits = (
            dice_layer(Tensor(ind_data_dict["valid features"]).to(device)).cpu().numpy()
        )
    ind_energy = logsumexp(ind_valid_logits, axis=1)
    ind_energy = np.asarray(ind_energy)
    ind_data_dict["dice"] = ind_energy
    # OoD
    for ood_name in ood_names:
        with torch.no_grad():
            ood_logits = (
                dice_layer(Tensor(ood_data_dict[f"{ood_name} features"]).to(device)).cpu().numpy()
            )
        ood_energy = logsumexp(ood_logits, axis=1)
        ood_baselines_dict[f"{ood_name} dice"] = np.asarray(ood_energy)

    return ind_data_dict, ood_baselines_dict


def get_react_score_from_features(
    fc_params: Dict[str, np.ndarray],
    ind_data_dict: Dict[str, np.ndarray],
    ood_data_dict: Dict[str, np.ndarray],
    ood_names: List[str],
    ood_baselines_dict: Dict[str, np.ndarray],
    percentile: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute ReAct scores by clipping activations and computing energy.

    The ReAct method clips feature activations at a percentile threshold computed
    from InD training features and then computes the energy score (log-sum-exp)
    from the resulting logits.

    Args:
        fc_params: Dictionary containing the final linear layer parameters with keys
            "weight" and "bias".
        ind_data_dict: Dictionary with InD data arrays. Expected keys include
            "train features" and "valid features".
        ood_data_dict: Dictionary with OoD data arrays. Each OoD dataset should
            provide a "{ood_name} features" entry.
        ood_names: List of OoD dataset names present in `ood_data_dict`.
        ood_baselines_dict: Dictionary that will be populated with computed OoD
            baseline scores (keyed by "{ood_name} react").
        percentile: Percentile value used to compute the activation clipping
            threshold from the flattened InD training features.

    Returns:
        Tuple of (updated ind_data_dict, updated ood_baselines_dict) where the
        InD dictionary receives a "react" entry and the OoD baselines dictionary
        receives entries for each OoD dataset under the key "{ood_name} react".
    """
    print("Calculating ReAct score")
    w, b = fc_params["weight"], fc_params["bias"]
    if isinstance(w, Tensor):
        w = w.numpy()
    if isinstance(b, Tensor):
        b = b.numpy()
    # Calculate threshold
    activation_threshold = np.percentile(ind_data_dict["train features"].flatten(), percentile)
    clipped_ind_valid_features = ind_data_dict["valid features"].clip(max=activation_threshold)
    ind_valid_logits = np.matmul(clipped_ind_valid_features, w.T) + b
    ind_energy = logsumexp(ind_valid_logits, axis=1)
    ind_energy = np.asarray(ind_energy)
    ind_data_dict["react"] = ind_energy
    # OoD
    for ood_name in ood_names:
        ood_clipped_features = ood_data_dict[f"{ood_name} features"].clip(max=activation_threshold)
        ood_logits = np.matmul(ood_clipped_features, w.T) + b
        ood_energy = logsumexp(ood_logits, axis=1)
        ood_baselines_dict[f"{ood_name} react"] = np.asarray(ood_energy)

    return ind_data_dict, ood_baselines_dict


def get_dice_react_score_from_features(
    fc_params: Dict[str, np.ndarray],
    ind_data_dict: Dict[str, np.ndarray],
    ood_data_dict: Dict[str, np.ndarray],
    ood_names: List[str],
    ood_baselines_dict: Dict[str, np.ndarray],
    dice_percentile: int,
    react_percentile: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Combine DICE and ReAct: clip activations then apply a RouteDICE layer.

    This method first applies an activation clipping similar to ReAct and then
    computes logits via a RouteDICE layer. The energy score (log-sum-exp) is
    computed on the resulting logits for both InD validation and OoD datasets.

    Args:
        fc_params: Dictionary containing the final linear layer parameters with keys
            "weight" and "bias".
        ind_data_dict: Dictionary with InD data arrays. Expected keys include
            "train features", "valid features", and "train logits".
        ood_data_dict: Dictionary with OoD data arrays. Each OoD dataset should
            provide a "{ood_name} features" entry.
        ood_names: List of OoD dataset names present in `ood_data_dict`.
        ood_baselines_dict: Dictionary that will be populated with computed OoD
            baseline scores (keyed by "{ood_name} dice_react").
        dice_percentile: Percentile used by RouteDICE for routing.
        react_percentile: Percentile used to compute clipping threshold for ReAct.

    Returns:
        Tuple of (updated ind_data_dict, updated ood_baselines_dict) where the
        InD dictionary receives a "dice_react" entry and the OoD baselines dictionary
        receives entries for each OoD dataset under the key "{ood_name} dice_react".
    """
    print("Calculating DICE+ReAct score")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(fc_params["weight"], np.ndarray) or isinstance(fc_params["bias"], np.ndarray):
        params_tensor = {"weight": Tensor(fc_params["weight"]), "bias": Tensor(fc_params["bias"])}
    else:
        params_tensor = {"weight": fc_params["weight"], "bias": fc_params["bias"]}
    # DICE Get mean per feature dimension
    dice_info = Tensor(ind_data_dict["train features"]).mean(0).cpu().numpy()
    dice_layer = RouteDICE(
        in_features=ind_data_dict["train features"].shape[1],
        out_features=ind_data_dict["train logits"].shape[1],
        bias=True,
        p=dice_percentile,
        info=dice_info,
    )
    dice_layer.load_state_dict(params_tensor)
    dice_layer.to(device)
    dice_layer.eval()

    # React Calculate threshold
    activation_threshold = np.percentile(
        ind_data_dict["train features"].flatten(), react_percentile
    )
    # Valid set scores
    clipped_ind_valid_features = ind_data_dict["valid features"].clip(max=activation_threshold)
    with torch.no_grad():
        ind_valid_logits = dice_layer(Tensor(clipped_ind_valid_features).to(device)).cpu().numpy()
    ind_energy = logsumexp(ind_valid_logits, axis=1)
    ind_energy = np.asarray(ind_energy)
    ind_data_dict["dice_react"] = ind_energy
    # OoD
    for ood_name in ood_names:
        ood_clipped_features = ood_data_dict[f"{ood_name} features"].clip(max=activation_threshold)
        with torch.no_grad():
            ood_logits = dice_layer(Tensor(ood_clipped_features).to(device)).cpu().numpy()
        ood_energy = logsumexp(ood_logits, axis=1)
        ood_baselines_dict[f"{ood_name} dice_react"] = np.asarray(ood_energy)

    return ind_data_dict, ood_baselines_dict


# ASH with scaling for convolutional layers
def ash_s_conv_layer(x: Tensor, percentile: int = 65):
    """Apply ASH-S scaling for convolutional feature maps.

    This function implements the ASH-S pruning and scaling operation for 4D
    convolutional tensors (B, C, H, W). It zeros out lower-importance elements
    per sample and rescales the remaining values to preserve global energy.

    Args:
        x: A 4D torch Tensor with shape (batch, channels, height, width).
        percentile: Percentile used to determine how many elements to keep.

    Returns:
        The sharpened tensor with the same shape as the input.
    """
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    # calculate new sum of the input per sample after pruning
    s2 = x.sum(dim=[1, 2, 3])

    # apply sharpening
    scale = s1 / s2
    x = x * torch.exp(scale[:, None, None, None])

    return x


def ash_s_linear_layer(x: np.ndarray, percentile: int = 85):
    """Apply ASH-S scaling for linear (2D) activations.

    This function keeps the top-k elements per row of a 2D numpy array and
    applies a scaling similar to the convolutional version.

    Args:
        x: 2D numpy array with shape (batch, features).
        percentile: Percentile used to determine how many features to keep.

    Returns:
        A 2D numpy array with the pruned-and-scaled activations.
    """
    assert x.ndim == 2
    assert 0 <= percentile <= 100
    # calculate the sum of the input per sample
    s1 = x.sum(axis=1)
    n = x.shape[1]
    k = n - int(np.round(n * percentile / 100.0))
    idx = np.argpartition(x, -k)[:, -k:]
    top_k = np.partition(x, -k)[:, -k:]
    scattered = np.zeros_like(x)
    np.put_along_axis(scattered, indices=idx, values=top_k, axis=1)

    # calculate new sum of the input per sample after pruning
    s2 = scattered.sum(axis=1)

    # apply sharpening
    scale = s1 / s2
    scattered = scattered * np.exp(scale[:, None])

    return scattered


def get_ash_score_from_features(
    fc_params: Dict[str, np.ndarray],
    ind_data_dict: Dict[str, np.ndarray],
    ood_data_dict: Dict[str, np.ndarray],
    ood_names: List[str],
    ood_baselines_dict: Dict[str, np.ndarray],
    ash_percentile: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute ASH-based energy scores from linear features.

    The function applies the ASH-S linear pruning to the validation features,
    computes logits using provided FC parameters and returns the energy score for
    InD and OoD datasets.

    Args:
        fc_params: Dictionary containing the final linear layer parameters with keys
            "weight" and "bias".
        ind_data_dict: Dictionary with InD data arrays. Expected keys include
            "valid features".
        ood_data_dict: Dictionary with OoD data arrays. Each OoD dataset should
            provide a "{ood_name} features" entry.
        ood_names: List of OoD dataset names present in `ood_data_dict`.
        ood_baselines_dict: Dictionary that will be populated with computed OoD
            baseline scores (keyed by "{ood_name} ash").
        ash_percentile: Percentile used in ASH-S linear pruning.

    Returns:
        Tuple of (updated ind_data_dict, updated ood_baselines_dict) where the
        InD dictionary receives an "ash" entry and the OoD baselines dictionary
        receives entries for each OoD dataset under the key "{ood_name} ash".
    """
    print("Calculating ash score")
    w, b = fc_params["weight"], fc_params["bias"]
    if isinstance(w, Tensor):
        w = w.numpy()
    if isinstance(b, Tensor):
        b = b.numpy()
    ind_valid_scattered_features = ash_s_linear_layer(
        ind_data_dict["valid features"], ash_percentile
    )
    ind_valid_logits = np.matmul(ind_valid_scattered_features, w.T) + b
    ind_energy = logsumexp(ind_valid_logits, axis=1)
    ind_energy = np.asarray(ind_energy)
    ind_data_dict["ash"] = ind_energy
    # OoD
    for ood_name in ood_names:
        ood_scattered_features = ash_s_linear_layer(
            ood_data_dict[f"{ood_name} features"], ash_percentile
        )
        ood_logits = np.matmul(ood_scattered_features, w.T) + b
        ood_energy = logsumexp(ood_logits, axis=1)
        ood_baselines_dict[f"{ood_name} ash"] = np.asarray(ood_energy)

    return ind_data_dict, ood_baselines_dict


def generalized_entropy(probs, gamma, M):
    """
    Calculates a generalized entropy score based on the top M probabilities.

    This function determines the generalized entropy for a set of probability
    distributions. It focuses on the top M probabilities (sorted in descending
    order) and applies a generalized entropy formula with a specific gamma value
    to emphasize concentration or dispersion.

    Args:
        probs: ndarray
            A 2D array where each row represents a probability distribution.
        gamma: float
            A parameter used to adjust the sensitivity to concentration within
            the probability distribution. Higher values of gamma result in
            stronger emphasis on higher probabilities.
        M: int
            The number of top probabilities from each distribution to consider in
            the computation.

    Returns:
        ndarray:
            An array containing the negative generalized entropy scores for each
            distribution row in the input.
    """
    probs_sorted = np.sort(probs, axis=1)[:, -M:]
    scores = np.sum(probs_sorted**gamma * (1 - probs_sorted) ** (gamma), axis=1)

    return -scores


def get_gen_score_from_logits(
    ind_data_dict: Dict[str, np.ndarray],
    ood_data_dict: Dict[str, np.ndarray],
    ood_names: List[str],
    ood_baselines_dict: Dict[str, np.ndarray],
    gamma: float,
    gen_m: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute generalized-entropy (GEN) score from logits.

    The GEN baseline computes a score that emphasizes the top-M probabilities in
    the softmax distribution and applies a gamma exponent to accentuate
    concentration.

    Args:
        ind_data_dict: Dictionary with InD data arrays. Expected key "valid logits".
        ood_data_dict: Dictionary with OoD data arrays. Each OoD dataset should
            provide a "{ood_name} logits" entry.
        ood_names: List of OoD dataset names present in `ood_data_dict`.
        ood_baselines_dict: Dictionary that will be populated with computed OoD
            baseline scores (keyed by "{ood_name} gen").
        gamma: Exponent parameter for generalized entropy.
        gen_m: Number of top probabilities (M) to consider per sample.

    Returns:
        Tuple of (updated ind_data_dict, updated ood_baselines_dict) where the
        InD dictionary receives a "gen" entry and the OoD baselines dictionary
        receives entries for each OoD dataset under the key "{ood_name} gen".
    """
    print("Calculating GEN score")

    softmax_ind_valid = softmax(ind_data_dict["valid logits"], axis=1)
    gen_score_ind_valid = generalized_entropy(softmax_ind_valid, gamma, gen_m)
    ind_data_dict["gen"] = gen_score_ind_valid
    for ood_name in ood_names:
        softmax_ood = softmax(ood_data_dict[f"{ood_name} logits"], axis=1)
        gen_score_ood = generalized_entropy(softmax_ood, gamma, gen_m)
        ood_baselines_dict[f"{ood_name} gen"] = gen_score_ood

    return ind_data_dict, ood_baselines_dict


def calculate_vim_score(
    fc_params: Dict[str, np.ndarray],
    ind_data_dict: Dict[str, np.ndarray],
    ood_data_dict: Dict[str, np.ndarray],
    ood_names: List[str],
    ood_baselines_dict: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute the ViM score for InD and OoD samples.

    ViM combines a projection-based novelty measure with the energy of logits.
    This function computes per-sample ViM scores for the validation set and for
    each OoD dataset and stores them in the provided dictionaries.

    Args:
        fc_params: Dictionary containing the final linear layer parameters with keys
            "weight" and "bias".
        ind_data_dict: Dictionary with InD data arrays. Expected keys include
            "train features", "valid features", "train logits", "valid logits".
        ood_data_dict: Dictionary with OoD data arrays. Each OoD dataset should
            provide "{ood_name} features" and "{ood_name} logits" entries.
        ood_names: List of OoD dataset names present in `ood_data_dict`.
        ood_baselines_dict: Dictionary that will be populated with computed OoD
            baseline scores (keyed by "{ood_name} vim").

    Returns:
        Tuple of (updated ind_data_dict, updated ood_baselines_dict) where the
        InD dictionary receives a "vim" entry and the OoD baselines dictionary
        receives entries for each OoD dataset under the key "{ood_name} vim".
    """
    print("Calculating ViM score")
    w, b = fc_params["weight"], fc_params["bias"]
    if isinstance(w, Tensor):
        w = w.numpy()
    if isinstance(b, Tensor):
        b = b.numpy()
    u = -np.matmul(np.linalg.pinv(w), b)

    if ind_data_dict["train features"].shape[-1] >= 2048:
        DIM = 1000
    elif ind_data_dict["train features"].shape[-1] >= 768:
        DIM = 512
    else:
        DIM = ind_data_dict["train features"].shape[-1] // 2
    print(f"{DIM=}")
    feature_id_train = ind_data_dict["train features"]
    feature_id_val = ind_data_dict["valid features"]
    # Last class is the background
    logit_id_train = ind_data_dict["train logits"]
    logit_id_val = ind_data_dict["valid logits"]

    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feature_id_train - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

    vlogit_id_train = np.linalg.norm(np.matmul(feature_id_train - u, NS), axis=-1)
    alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
    print(f"{alpha=:.4f}")

    vlogit_id_val = np.linalg.norm(np.matmul(feature_id_val - u, NS), axis=-1) * alpha
    energy_id_val = logsumexp(logit_id_val, axis=-1)
    score_id = -vlogit_id_val + energy_id_val
    ind_data_dict["vim"] = score_id

    for ood_name in ood_names:
        logit_ood = ood_data_dict[f"{ood_name} logits"]
        feature_ood = ood_data_dict[f"{ood_name} features"]
        energy_ood = logsumexp(logit_ood, axis=-1)
        vlogit_ood = np.linalg.norm(np.matmul(feature_ood - u, NS), axis=-1) * alpha
        score_ood = -vlogit_ood + energy_ood
        ood_baselines_dict[f"{ood_name} vim"] = score_ood

    return ind_data_dict, ood_baselines_dict


def get_msp_score_from_logits(
    ind_data_dict: Dict[str, np.ndarray],
    ood_data_dict: Dict[str, np.ndarray],
    ood_names: List[str],
    ood_baselines_dict: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute maximum softmax probability (MSP) baseline score.

    MSP is computed as the maximum probability from the softmax of logits. The
    function computes MSP for the InD validation set and for each OoD dataset.

    Args:
        ind_data_dict: Dictionary with InD data arrays. Expected key "valid logits".
        ood_data_dict: Dictionary with OoD data arrays. Each OoD dataset should
            provide a "{ood_name} logits" entry.
        ood_names: List of OoD dataset names present in `ood_data_dict`.
        ood_baselines_dict: Dictionary that will be populated with computed OoD
            baseline scores (keyed by "{ood_name} msp").

    Returns:
        Tuple of (updated ind_data_dict, updated ood_baselines_dict) where the
        InD dictionary receives a "msp" entry and the OoD baselines dictionary
        receives entries for each OoD dataset under the key "{ood_name} msp".
    """
    print("Calculating msp score")
    ind_valid_msp = np.max(softmax(ind_data_dict["valid logits"], axis=1), axis=1)
    ind_data_dict["msp"] = ind_valid_msp
    for ood_name in ood_names:
        ood_msp = np.max(softmax(ood_data_dict[f"{ood_name} logits"], axis=1), axis=1)
        ood_baselines_dict[f"{ood_name} msp"] = ood_msp

    return ind_data_dict, ood_baselines_dict


def get_raw_score_from_logits(
    ind_data_dict: Dict[str, np.ndarray],
    ood_data_dict: Dict[str, np.ndarray],
    ood_names: List[str],
    ood_baselines_dict: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute raw predictions without postprocessing for OOD detection.

    Analog to msp, except the threshold is set to 1.0 so that no OOD correction is done

    Args:
        ind_data_dict: Dictionary with InD data arrays. Expected key "valid logits".
        ood_data_dict: Dictionary with OoD data arrays. Each OoD dataset should
            provide a "{ood_name} logits" entry.
        ood_names: List of OoD dataset names present in `ood_data_dict`.
        ood_baselines_dict: Dictionary that will be populated with computed OoD
            baseline scores (keyed by "{ood_name} msp").

    Returns:
        Tuple of (updated ind_data_dict, updated ood_baselines_dict) where the
        InD dictionary receives a "raw" entry and the OoD baselines dictionary
        receives entries for each OoD dataset under the key "{ood_name} raw".
    """
    print("Calculating raw score")
    ind_valid_msp = np.max(softmax(ind_data_dict["valid logits"], axis=1), axis=1)
    ind_data_dict["raw"] = ind_valid_msp
    for ood_name in ood_names:
        ood_msp = np.max(softmax(ood_data_dict[f"{ood_name} logits"], axis=1), axis=1)
        ood_baselines_dict[f"{ood_name} raw"] = ood_msp

    return ind_data_dict, ood_baselines_dict


def get_energy_score_from_logits(
    ind_data_dict: Dict[str, np.ndarray],
    ood_data_dict: Dict[str, np.ndarray],
    ood_names: List[str],
    ood_baselines_dict: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute the energy baseline score from logits.

    Energy is computed as log-sum-exp across logits for each sample. The
    function populates the InD and OoD dictionaries with the energy scores.

    Args:
        ind_data_dict: Dictionary with InD data arrays. Expected key "valid logits".
        ood_data_dict: Dictionary with OoD data arrays. Each OoD dataset should
            provide a "{ood_name} logits" entry.
        ood_names: List of OoD dataset names present in `ood_data_dict`.
        ood_baselines_dict: Dictionary that will be populated with computed OoD
            baseline scores (keyed by "{ood_name} energy").

    Returns:
        Tuple of (updated ind_data_dict, updated ood_baselines_dict) where the
        InD dictionary receives an "energy" entry and the OoD baselines dictionary
        receives entries for each OoD dataset under the key "{ood_name} energy".
    """
    print("Calculating energy score")
    ind_energy = logsumexp(ind_data_dict["valid logits"], axis=1)
    ind_energy = np.asarray(ind_energy)
    ind_data_dict["energy"] = ind_energy
    for ood_name in ood_names:
        ood_energy = logsumexp(ood_data_dict[f"{ood_name} logits"], axis=1)
        ood_baselines_dict[f"{ood_name} energy"] = np.asarray(ood_energy)

    return ind_data_dict, ood_baselines_dict


def get_mahalanobis_score_from_features(
    ind_data_dict: Dict[str, np.ndarray],
    ood_data_dict: Dict[str, np.ndarray],
    ood_names: List[str],
    ood_baselines_dict: Dict[str, np.ndarray],
    num_classes: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute Mahalanobis distance-based scores for InD and OoD samples.

    This function estimates class means and a shared precision matrix from InD
    training features and computes per-sample Mahalanobis-based confidence
    scores for the validation set and OoD datasets.

    Args:
        ind_data_dict: Dictionary with InD data arrays. Expected keys include
            "train features" and "valid features" and label keys.
        ood_data_dict: Dictionary with OoD data arrays. Each OoD dataset should
            provide a "{ood_name} features" entry.
        ood_names: List of OoD dataset names present in `ood_data_dict`.
        ood_baselines_dict: Dictionary that will be populated with computed OoD
            baseline scores (keyed by "{ood_name} mdist").
        num_classes: Number of classes used to compute class-wise statistics.

    Returns:
        Tuple of (updated ind_data_dict, updated ood_baselines_dict) where the
        InD dictionary receives an "mdist" entry and the OoD baselines dictionary
        receives entries for each OoD dataset under the key "{ood_name} mdist".
    """
    print("Calculating mahalanobis score")
    class_mean, precision = mahalanobis_preprocess(ind_data=ind_data_dict, num_classes=num_classes)
    ind_valid_score = mahalanobis_postprocess(
        feats=ind_data_dict["valid features"],
        class_mean=class_mean,
        precision=precision,
        num_classes=num_classes,
    )
    ind_data_dict["mdist"] = ind_valid_score
    for ood_name in ood_names:
        ood_score = mahalanobis_postprocess(
            feats=ood_data_dict[f"{ood_name} features"],
            class_mean=class_mean,
            precision=precision,
            num_classes=num_classes,
        )
        ood_baselines_dict[f"{ood_name} mdist"] = ood_score

    return ind_data_dict, ood_baselines_dict


def mahalanobis_preprocess(
    ind_data: Dict[str, np.ndarray], num_classes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate class means and precision (inverse covariance) for Mahalanobis.

    The function computes per-class means from the InD training features and
    fits an EmpiricalCovariance on the centered data to obtain a precision matrix
    for subsequent Mahalanobis distance computations.

    Args:
        ind_data: Dictionary with InD data arrays. Expects keys "train features"
            and "train labels".
        num_classes: Number of classes to compute statistics for.

    Returns:
        A tuple (class_mean, precision) where `class_mean` has shape
        [num_classes, feature_dim] and `precision` is the precision matrix
        returned by the fitted EmpiricalCovariance.
    """
    class_mean = []
    centered_data = []
    for c in range(num_classes):
        class_samples = ind_data["train features"][ind_data["train labels"] == c]
        if len(class_samples) == 0:
            warnings.warn(f"No train examples for class {c}")
        class_mean.append(class_samples.mean(0))
        centered_data.append(class_samples - class_mean[c].reshape(1, -1))
    class_mean = np.stack(class_mean)  # shape [#classes, feature dim]

    group_lasso = EmpiricalCovariance(assume_centered=False)

    group_lasso.fit(np.concatenate(centered_data).astype(np.float32))
    # inverse of covariance
    return class_mean, group_lasso.precision_


def mahalanobis_postprocess(
    feats: np.ndarray, class_mean: np.ndarray, precision: np.ndarray, num_classes: int
) -> np.ndarray:
    """Postprocess features into Mahalanobis-based confidence scores.

    For each sample, the function computes the Mahalanobis quadratic form with
    respect to each class mean and returns the maximum (best) confidence per
    sample. Classes without training examples are excluded by replacing NaNs
    with -inf.

    Args:
        feats: Array of shape (N, feature_dim) with features to score.
        class_mean: Array of per-class means with shape (num_classes, feature_dim).
        precision: Precision (inverse covariance) matrix used for the quadratic form.
        num_classes: Number of classes considered.

    Returns:
        1D numpy array of length N with the per-sample Mahalanobis confidence scores.
    """
    all_conf_score = []
    for feats in feats:
        class_scores = np.zeros((1, num_classes))
        for c in range(num_classes):
            tensor = feats - class_mean[c].reshape(1, -1)
            class_scores[:, c] = np.diag(-np.matmul(np.matmul(tensor, precision), tensor.T))
        # Exclude the score for classes with no examples in the training data!
        class_scores[np.isnan(class_scores)] = -np.inf
        conf = np.max(class_scores, axis=1)

        all_conf_score.append(conf)

    all_conf_score_t = np.concatenate(all_conf_score)

    return all_conf_score_t


def get_knn_score_from_features(
    ind_data_dict: Dict[str, np.ndarray],
    ood_data_dict: Dict[str, np.ndarray],
    ood_names: List[str],
    ood_baselines_dict: Dict[str, np.ndarray],
    k_neighbors: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute K-NN based novelty scores using Faiss.

    The function builds a Faiss index on the normalized training activations and
    returns, for each sample, the negative distance to the k-th nearest neighbor
    (higher indicates more in-distribution).

    Args:
        ind_data_dict: Dictionary with InD data arrays. Expected key "train features"
            and "valid features".
        ood_data_dict: Dictionary with OoD data arrays. Each OoD dataset should
            provide a "{ood_name} features" entry.
        ood_names: List of OoD dataset names present in `ood_data_dict`.
        ood_baselines_dict: Dictionary that will be populated with computed OoD
            baseline scores (keyed by "{ood_name} knn").
        k_neighbors: Number of neighbors to use in the KNN search.

    Returns:
        Tuple of (updated ind_data_dict, updated ood_baselines_dict) where the
        InD dictionary receives a "knn" entry and the OoD baselines dictionary
        receives entries for each OoD dataset under the key "{ood_name} knn".
    """
    print("Calculating knn score")
    # Prepare knn model with training data
    train_activations = (
        normalizer(ind_data_dict["train features"])
        if isinstance(ind_data_dict["train features"], np.ndarray)
        else np.asarray(normalizer(ind_data_dict["train features"]))
    )
    # Faiss requires float32 C-contiguous arrays
    train_activations = np.ascontiguousarray(train_activations.astype(np.float32))
    index = faiss.IndexFlatL2(ind_data_dict["train features"].shape[1])
    index.add(train_activations)  # type: ignore[arg-type]

    def postprocess_knn(data: np.ndarray, k: int) -> np.ndarray:
        k_scores = []
        for sample in data:
            activations = normalizer(sample.reshape(1, -1))
            activations = np.ascontiguousarray(np.asarray(activations).astype(np.float32))
            # search returns (distances, labels)
            # Explicitly cast k to int; add type ignore because the static analyzer
            # cannot infer faiss' C-extension signature correctly.
            D, _ = index.search(activations, int(k))  # type: ignore[call-arg]
            kth_dist = -D[:, -1]
            k_scores.append(kth_dist)
        return np.concatenate(k_scores, axis=0)

    # InD valid
    ind_data_dict["knn"] = postprocess_knn(ind_data_dict["valid features"], k_neighbors)

    # OoD datasets
    for ood_name in ood_names:
        ood_score = postprocess_knn(ood_data_dict[f"{ood_name} features"], k_neighbors)
        ood_baselines_dict[f"{ood_name} knn"] = ood_score

    return ind_data_dict, ood_baselines_dict


def get_labels_from_logits(
    id_data: Dict[str, np.ndarray], ood_data: Dict[str, np.ndarray], ood_names: List[str]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Extract predicted labels from logits for both ID and OoD sets.

    This helper inspects available logits arrays and converts them to integer
    predicted labels using argmax. It also handles special cases where logits
    are empty lists (interpreted as missing labels) and raises NotImplementedError
    if the inputs do not match expected forms.

    Args:
        id_data: Dictionary with ID logits under keys "train logits" and
            "valid logits" (or those keys absent/empty).
        ood_data: Dictionary with OoD logits under keys "{ood_name} logits".
        ood_names: List of OoD dataset names to process.

    Returns:
        Tuple of (updated id_data, updated ood_data) where each dictionary has
        additional keys "train labels" and "valid labels" (ID) and
        "{ood_name} labels" (OoD) populated as numpy arrays or empty lists.

    Raises:
        NotImplementedError: If the input logits are not numpy arrays or expected
            empty-list placeholders.
    """

    id_train_logits = id_data.pop("train logits", None) if "train logits" in id_data else None
    id_valid_logits = id_data.pop("valid logits", None) if "valid logits" in id_data else None
    # In the case logits were extracted:
    if isinstance(id_train_logits, np.ndarray) or isinstance(id_valid_logits, np.ndarray):
        if id_train_logits is not None:
            if id_train_logits.shape[1] == 21 or id_train_logits.shape[1] == 11:
                id_train_logits = id_train_logits[:, :-1]
            id_train_labels = np.argmax(id_train_logits, axis=-1)
        else:
            id_train_labels = np.asarray([], dtype=int)
        if id_valid_logits is not None:
            if id_valid_logits.shape[1] == 21 or id_valid_logits.shape[1] == 11:
                id_valid_logits = id_valid_logits[:, :-1]
            id_valid_labels = np.argmax(id_valid_logits, axis=-1)
        else:
            id_valid_labels = np.asarray([], dtype=int)

        id_data["train labels"] = id_train_labels
        id_data["valid labels"] = id_valid_labels
    # In case labels were not extracted
    elif (
        isinstance(id_train_logits, list)
        and len(id_train_logits) == 0
        and isinstance(id_valid_logits, list)
        and len(id_valid_logits) == 0
    ):
        id_data["train labels"] = np.asarray([], dtype=int)
        id_data["valid labels"] = np.asarray([], dtype=int)
    else:
        raise NotImplementedError

    for ood_name in ood_names:
        ood_logits = ood_data.pop(f"{ood_name} logits", None)
        if isinstance(ood_logits, np.ndarray):
            if ood_logits.shape[1] == 21 or ood_logits.shape[1] == 11:
                ood_logits = ood_logits[:, :-1]
            ood_labels = np.argmax(ood_logits, axis=-1)
            ood_data[f"{ood_name} labels"] = ood_labels
        elif isinstance(ood_logits, list) and len(ood_logits) == 0:
            ood_data[f"{ood_name} labels"] = np.asarray([], dtype=int)
        else:
            raise NotImplementedError

    return id_data, ood_data


def remove_latent_features(
    id_data: Dict[str, np.ndarray], ood_data: Dict[str, np.ndarray], ood_names: List[str]
):
    """Remove feature arrays from id and ood dictionaries.

    This utility removes keys related to latent features (train/valid features)
    from the provided dictionaries to free memory or avoid persisting large
    arrays when only logits/labels are needed.

    Args:
        id_data: Dictionary containing potential "train features" and "valid features".
        ood_data: Dictionary containing potential "{ood_name} features" entries.
        ood_names: List of OoD dataset names whose feature keys should be removed.

    Returns:
        Tuple of (id_data, ood_data) after removing feature entries. The function
        silently ignores missing keys.
    """

    id_data.pop("train features", None)
    id_data.pop("valid features", None)
    for ood_name in ood_names:
        ood_data.pop(f"{ood_name} features", None)

    return id_data, ood_data


# DDU calculations
def gmm_fit(
    embeddings: Tensor, labels: Tensor, num_classes: int
) -> Tuple[torch.distributions.MultivariateNormal, float]:
    """Fit a class-conditional Gaussian Mixture approximation (per-class MVN).

    The function computes per-class empirical means and covariance matrices
    (with handling for classes that have no examples). It then attempts to build
    a torch.distributions.MultivariateNormal object for the collection of
    class means and covariances, adding a small jitter if necessary to ensure
    positive-definiteness.

    Args:
        embeddings: 2D torch Tensor of shape (N, D) containing feature embeddings.
        labels: 1D torch Tensor with integer class labels for the embeddings.
        num_classes: Number of classes expected.

    Returns:
        A tuple (gmm, jitter_eps) where `gmm` is a torch.distributions.MultivariateNormal
        instance parameterized by stacked class means and covariance matrices,
        and `jitter_eps` is the last jitter scalar tried (0 or a small power of 10)
        that permitted successful construction.
    """
    jitters = [0] + [10**exp for exp in range(-20, 0, 1)]

    def centered_cov_torch(x):
        n = x.shape[0]
        if n == 1:
            n += 1
        res = 1 / (n - 1) * x.t().mm(x)
        return res

    with torch.no_grad():
        classwise_mean_features = torch.stack(
            [torch.mean(embeddings[labels == c], dim=0) for c in range(num_classes)]
        )
        classwise_cov_features = torch.stack(
            [
                centered_cov_torch(embeddings[labels == c] - classwise_mean_features[c])
                for c in range(num_classes)
            ]
        )
    # Control for classes with no examples
    n_cols_to_remove = torch.any(classwise_mean_features.isnan(), dim=1, keepdim=True).sum().item()
    if n_cols_to_remove > 0:
        cols_bool = ~torch.any(classwise_mean_features.isnan(), dim=1, keepdim=True)
        remaining_cols = classwise_mean_features.shape[0] - n_cols_to_remove
        hidden_dim = classwise_mean_features.shape[1]
        # Subset means
        mean_mask = cols_bool.repeat(1, hidden_dim)
        classwise_mean_features = classwise_mean_features[mean_mask].reshape(remaining_cols, -1)
        # Subset covariances
        cov_mask = (
            cols_bool.repeat(1, hidden_dim)
            .repeat(1, hidden_dim)
            .reshape(num_classes, hidden_dim, hidden_dim)
        )
        classwise_cov_features = classwise_cov_features[cov_mask].reshape(
            remaining_cols, hidden_dim, hidden_dim
        )

    with torch.no_grad():
        for jitter_eps in jitters:
            try:
                jitter = jitter_eps * torch.eye(
                    classwise_cov_features.shape[1],
                    device=classwise_cov_features.device,
                ).unsqueeze(0)
                gmm = torch.distributions.MultivariateNormal(
                    loc=classwise_mean_features,
                    covariance_matrix=(classwise_cov_features + jitter),
                )
            except RuntimeError as e:
                if "cholesky" in str(e):
                    continue
            except ValueError as e:
                if "found invalid values" in str(e):
                    continue
            break

    return gmm, jitter_eps


def get_ddu_score_from_features(
    ind_data_dict: Dict[str, np.ndarray],
    ood_data_dict: Dict[str, np.ndarray],
    ood_names: List[str],
    ood_baselines_dict: Dict[str, np.ndarray],
    num_classes: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute DDU scores by fitting per-class Gaussians and scoring log-prob.

    DDU (Deep Deterministic Uncertainty) leverages a fitted per-class Gaussian
    to compute per-sample log-probabilities; the function aggregates class log
    probabilities and stores the energy-like score for InD validation and each
    OoD dataset.

    Args:
        ind_data_dict: Dictionary with InD data arrays. Expected keys include
            "train features" and "valid features" and "train labels".
        ood_data_dict: Dictionary with OoD data arrays. Each OoD dataset should
            provide a "{ood_name} features" entry.
        ood_names: List of OoD dataset names present in `ood_data_dict`.
        ood_baselines_dict: Dictionary that will be populated with computed OoD
            baseline scores (keyed by "{ood_name} ddu").
        num_classes: Number of classes used to fit the per-class Gaussians.

    Returns:
        Tuple of (updated ind_data_dict, updated ood_baselines_dict) where the
        InD dictionary receives a "ddu" entry and the OoD baselines dictionary
        receives entries for each OoD dataset under the key "{ood_name} ddu".
    """
    print("Calculating ddu score")
    gmm, _ = gmm_fit(
        embeddings=Tensor(ind_data_dict["train features"]),
        labels=Tensor(ind_data_dict["train labels"]),
        num_classes=num_classes,
    )
    ddu_valid_log_probs = gmm.log_prob(Tensor(ind_data_dict["valid features"][:, None, :]))
    ind_energy = logsumexp(ddu_valid_log_probs.numpy(), axis=1)
    ind_energy = np.asarray(ind_energy)
    ind_data_dict["ddu"] = ind_energy
    for ood_name in ood_names:
        ood_log_probs = gmm.log_prob(Tensor(ood_data_dict[f"{ood_name} features"][:, None, :]))
        ood_energy = logsumexp(ood_log_probs.numpy(), axis=1)
        ood_baselines_dict[f"{ood_name} ddu"] = np.asarray(ood_energy)

    return ind_data_dict, ood_baselines_dict


def get_baselines_thresholds(
    baselines_names: List[str],
    baselines_scores_dict: Dict[str, np.ndarray],
    z_score_percentile: float = 1.645,
) -> Dict[str, float]:
    """Compute thresholds for baselines using a z-score-like percentile.

    For each baseline name provided, the function computes mean and standard
    deviation of the corresponding scores and returns a threshold defined as
    mean - z_score_percentile * std (useful when higher scores indicate InD).

    Args:
        baselines_names: List of baseline keys to compute thresholds for.
        baselines_scores_dict: Dictionary mapping baseline keys to numpy arrays
            with per-sample scores.
        z_score_percentile: Scalar multiplier corresponding to z-score cutoff
            (default 1.645 approximates the one-sided 95% quantile).

    Returns:
        Dictionary mapping baseline name to threshold float.
    """
    thresholds = {}
    for baseline_name in baselines_names:
        # Raw means no postprocessing; therefore, the threshold is zero so that no prediction gets corrected.
        # This is useful for calculating base model statistics on open set benchmarks
        if baseline_name == "raw":
            thresholds[baseline_name] = 0.0
        else:
            # Force numeric scalars
            mean = float(np.mean(baselines_scores_dict[baseline_name]))
            std = float(np.std(baselines_scores_dict[baseline_name]))
            # We suppose higher is ID, then 95% of ID scores should be above the threshold
            thresholds[baseline_name] = mean - (z_score_percentile * std)
    return thresholds


def calculate_all_baselines(
    baselines_names: List[str],
    ind_data_dict: Dict[str, np.ndarray],
    ood_data_dict: Dict[str, np.ndarray],
    fc_params: Union[Dict[str, np.ndarray], None],
    cfg: DictConfig,
    num_classes: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute a set of selected baseline OOD scores from available precomputed data.

    This convenience wrapper runs the baselines specified in `baselines_names`
    using the provided InD and OoD dictionaries and optional fc layer parameters.
    It returns updated dictionaries and a dictionary of computed OoD baseline
    scores keyed by "{ood_name} {baseline}".

    Args:
        baselines_names: List of baseline names to compute (e.g. 'vim','msp','knn').
        ind_data_dict: Dictionary containing InD data arrays such as "train features",
            "valid features", "train logits", and potentially "train labels".
        ood_data_dict: Dictionary containing OoD data arrays for each dataset.
        fc_params: Optional dictionary with fully-connected layer parameters if
            required by some baselines (weight and bias).
        cfg: Configuration object expected to contain baseline-specific hyperparams
            such as `k_neighbors`, `ash_percentile`, `gen_gamma`, `react_percentile`,
            and `dice_percentile`, and list `ood_datasets`.
        num_classes: Number of classes relevant for methods that require it.

    Returns:
        A tuple (ind_data_dict, ood_data_dict, ood_baselines_scores_dict) where
        the latter maps keys like "{ood_name} {baseline}" to numpy arrays of scores.
    """
    if num_classes > 21 and "gen" in baselines_names:
        raise ValueError(
            "Implementation of gen baseline does not yet support num_classes greater than 21. "
            "Otherwise implement M parameter specification"
        )
    ood_baselines_scores_dict = {}
    if "vim" in baselines_names:
        ind_data_dict, ood_baselines_scores_dict = calculate_vim_score(
            fc_params=fc_params,
            ind_data_dict=ind_data_dict,
            ood_data_dict=ood_data_dict,
            ood_names=cfg.ood_datasets,
            ood_baselines_dict=ood_baselines_scores_dict,
        )
    if "msp" in baselines_names:
        ind_data_dict, ood_baselines_scores_dict = get_msp_score_from_logits(
            ind_data_dict=ind_data_dict,
            ood_data_dict=ood_data_dict,
            ood_names=cfg.ood_datasets,
            ood_baselines_dict=ood_baselines_scores_dict,
        )
    if "raw" in baselines_names:
        ind_data_dict, ood_baselines_scores_dict = get_raw_score_from_logits(
            ind_data_dict=ind_data_dict,
            ood_data_dict=ood_data_dict,
            ood_names=cfg.ood_datasets,
            ood_baselines_dict=ood_baselines_scores_dict,
        )
    if "knn" in baselines_names:
        ind_data_dict, ood_baselines_scores_dict = get_knn_score_from_features(
            ind_data_dict=ind_data_dict,
            ood_data_dict=ood_data_dict,
            ood_names=cfg.ood_datasets,
            ood_baselines_dict=ood_baselines_scores_dict,
            k_neighbors=cfg.k_neighbors,
        )
    if "energy" in baselines_names:
        ind_data_dict, ood_baselines_scores_dict = get_energy_score_from_logits(
            ind_data_dict=ind_data_dict,
            ood_data_dict=ood_data_dict,
            ood_names=cfg.ood_datasets,
            ood_baselines_dict=ood_baselines_scores_dict,
        )
    if "ash" in baselines_names:
        ind_data_dict, ood_baselines_scores_dict = get_ash_score_from_features(
            fc_params=fc_params,
            ind_data_dict=ind_data_dict,
            ood_data_dict=ood_data_dict,
            ood_names=cfg.ood_datasets,
            ood_baselines_dict=ood_baselines_scores_dict,
            ash_percentile=cfg.ash_percentile,
        )
    if "gen" in baselines_names:
        ind_data_dict, ood_baselines_scores_dict = get_gen_score_from_logits(
            ind_data_dict=ind_data_dict,
            ood_data_dict=ood_data_dict,
            ood_names=cfg.ood_datasets,
            ood_baselines_dict=ood_baselines_scores_dict,
            gamma=cfg.gen_gamma,
            gen_m=num_classes,
        )
    if "react" in baselines_names:
        ind_data_dict, ood_baselines_scores_dict = get_react_score_from_features(
            fc_params=fc_params,
            ind_data_dict=ind_data_dict,
            ood_data_dict=ood_data_dict,
            ood_names=cfg.ood_datasets,
            ood_baselines_dict=ood_baselines_scores_dict,
            percentile=cfg.react_percentile,
        )
    if "dice" in baselines_names:
        ind_data_dict, ood_baselines_scores_dict = get_dice_score_from_features(
            fc_params=fc_params,
            ind_data_dict=ind_data_dict,
            ood_data_dict=ood_data_dict,
            ood_names=cfg.ood_datasets,
            ood_baselines_dict=ood_baselines_scores_dict,
            percentile=cfg.dice_percentile,
        )
    if "dice_react" in baselines_names:
        ind_data_dict, ood_baselines_scores_dict = get_dice_react_score_from_features(
            fc_params=fc_params,
            ind_data_dict=ind_data_dict,
            ood_data_dict=ood_data_dict,
            ood_names=cfg.ood_datasets,
            ood_baselines_dict=ood_baselines_scores_dict,
            dice_percentile=cfg.dice_percentile,
            react_percentile=cfg.react_percentile,
        )

    ind_data_dict, ood_data_dict = get_labels_from_logits(
        id_data=ind_data_dict, ood_data=ood_data_dict, ood_names=cfg.ood_datasets
    )
    if "mdist" in baselines_names:
        ind_data_dict, ood_baselines_scores_dict = get_mahalanobis_score_from_features(
            ind_data_dict=ind_data_dict,
            ood_data_dict=ood_data_dict,
            ood_names=cfg.ood_datasets,
            ood_baselines_dict=ood_baselines_scores_dict,
            num_classes=num_classes,
        )
    if "ddu" in baselines_names:
        ind_data_dict, ood_baselines_scores_dict = get_ddu_score_from_features(
            ind_data_dict=ind_data_dict,
            ood_data_dict=ood_data_dict,
            ood_names=cfg.ood_datasets,
            ood_baselines_dict=ood_baselines_scores_dict,
            num_classes=num_classes,
        )

    return ind_data_dict, ood_data_dict, ood_baselines_scores_dict
