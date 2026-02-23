# (c) 2023, CEA LIST
#
# All rights reserved.
# SPDX-License-Identifier: MIT
#
# Contributors
#    Based on https://github.com/fregu856/deeplabv3
#    Fabio Arnez, probabilistic adaptation
#    Daniel Montoya
from omegaconf import DictConfig
from scipy.special import softmax
from typing import Tuple, Dict, List, Union
import numpy as np

from runia_core.inference.postprocessors import DICE, ReAct, ASH, GEN, ViM, MSP, Energy, Mahalanobis, KNN, DDU, DICEReAct

__all__ = [
    "remove_latent_features",
    "calculate_all_baselines",
    "get_labels_from_logits",
    "baseline_name_dict"
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
    # Instantiate Postprocessor
    postp = DICE(
        flip_sign=False,
        dice_percentile=percentile,
        num_classes=ind_data_dict["train logits"].shape[1]
    )
    postp.setup(
        ind_train_data=ind_data_dict["train features"],
        valid_feats=ind_data_dict["valid features"],
        final_linear_layer_params=fc_params,
    )

    # Valid set scores
    ind_data_dict["dice"] = postp.postprocess(test_data=ind_data_dict["valid features"])
    # OoD
    for ood_name in ood_names:
        ood_baselines_dict[f"{ood_name} dice"] = postp.postprocess(
            test_data=ood_data_dict[f"{ood_name} features"]
        )

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
    postp = ReAct(
        flip_sign=False,
        react_percentile=percentile,
    )
    postp.setup(
        ind_train_data=ind_data_dict["train features"],
        valid_feats=ind_data_dict["valid features"],
        final_linear_layer_params=fc_params,
    )
    # Valid set scores
    ind_data_dict["react"] = postp.postprocess(test_data=ind_data_dict["valid features"])
    # OoD
    for ood_name in ood_names:
        ood_baselines_dict[f"{ood_name} react"] = postp.postprocess(
            test_data=ood_data_dict[f"{ood_name} features"]
        )

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
    postp = DICEReAct(
        flip_sign=False,
        dice_percentile=dice_percentile,
        react_percentile=react_percentile,
        num_classes=ind_data_dict["train logits"].shape[1]
    )
    postp.setup(
        ind_train_data=ind_data_dict["train features"],
        valid_feats=ind_data_dict["valid features"],
        final_linear_layer_params=fc_params,
    )

    ind_data_dict["dice_react"] = postp.postprocess(test_data=ind_data_dict["valid features"])
    # OoD
    for ood_name in ood_names:
        ood_baselines_dict[f"{ood_name} dice_react"] = postp.postprocess(test_data=ood_data_dict[f"{ood_name} features"])

    return ind_data_dict, ood_baselines_dict


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
    postp = ASH(
        flip_sign=False,
        ash_percentile=ash_percentile,
    )
    postp.setup(
        ind_train_data=ind_data_dict["train features"],
        valid_feats=ind_data_dict["valid features"],
        final_linear_layer_params=fc_params,
    )
    # InD valid scores
    ind_data_dict["ash"] = postp.postprocess(test_data=ind_data_dict["valid features"])
    # OoD
    for ood_name in ood_names:
        ood_baselines_dict[f"{ood_name} ash"] = postp.postprocess(test_data=ood_data_dict[f"{ood_name} features"])

    return ind_data_dict, ood_baselines_dict


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
    postp = GEN(
        flip_sign=False,
        gamma=gamma,
        num_classes=gen_m,
    )
    postp.setup(
        ind_train_data=ind_data_dict["train logits"],
    )
    ind_data_dict["gen"] = postp.postprocess(test_data=ind_data_dict["valid logits"])
    for ood_name in ood_names:
        ood_baselines_dict[f"{ood_name} gen"] = postp.postprocess(test_data=ood_data_dict[f"{ood_name} logits"])

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
    postp = ViM(
        flip_sign=False,
    )

    postp.setup(
        ind_train_data=ind_data_dict["train features"],
        train_logits=ind_data_dict["train logits"],
        valid_feats=ind_data_dict["valid features"],
        valid_logits=ind_data_dict["valid logits"],
        final_linear_layer_params=fc_params,
    )
    ind_data_dict["vim"] = postp.postprocess(
        test_data=ind_data_dict["valid features"], logits=ind_data_dict["valid logits"]
    )

    for ood_name in ood_names:
        ood_baselines_dict[f"{ood_name} vim"] = postp.postprocess(
            test_data=ood_data_dict[f"{ood_name} features"], logits=ood_data_dict[f"{ood_name} logits"]
        )

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
    postp = MSP(
        flip_sign=False,
    )
    postp.setup(
        ind_train_data=ind_data_dict["train logits"],
    )
    ind_data_dict["msp"] = postp.postprocess(test_data=ind_data_dict["valid logits"])
    for ood_name in ood_names:
        ood_baselines_dict[f"{ood_name} msp"] = postp.postprocess(test_data=ood_data_dict[f"{ood_name} logits"])

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
    postp = Energy(
        flip_sign=False,
    )
    postp.setup(
        ind_train_data=ind_data_dict["train logits"],
    )
    ind_data_dict["energy"] = postp.postprocess(test_data=ind_data_dict["valid logits"])
    for ood_name in ood_names:
        ood_baselines_dict[f"{ood_name} energy"] = postp.postprocess(test_data=ood_data_dict[f"{ood_name} logits"])

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
    postp = Mahalanobis(
        flip_sign=False,
        num_classes=num_classes,
    )
    postp.setup(
        ind_train_data=ind_data_dict["train features"],
        train_labels=ind_data_dict["train labels"],
        valid_feats=ind_data_dict["valid features"],
    )

    ind_data_dict["mdist"] = postp.postprocess(test_data=ind_data_dict["valid features"])
    for ood_name in ood_names:
        ood_baselines_dict[f"{ood_name} mdist"] = postp.postprocess(test_data=ood_data_dict[f"{ood_name} features"])

    return ind_data_dict, ood_baselines_dict


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
    postp = KNN(
        flip_sign=False,
        k_neighbors=k_neighbors,
    )
    postp.setup(
        ind_train_data=ind_data_dict["train features"],
        valid_feats=ind_data_dict["valid features"],
    )

    # InD valid
    ind_data_dict["knn"] = postp.postprocess(test_data=ind_data_dict["valid features"])

    # OoD datasets
    for ood_name in ood_names:
        ood_baselines_dict[f"{ood_name} knn"] = postp.postprocess(test_data=ood_data_dict[f"{ood_name} features"])

    return ind_data_dict, ood_baselines_dict


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
    postp = DDU(
        flip_sign=False,
        num_classes=num_classes,
    )
    postp.setup(
        ind_train_data=ind_data_dict["train features"],
        train_labels=ind_data_dict["train labels"],
        valid_feats=ind_data_dict["valid features"],
    )
    ind_data_dict["ddu"] = postp.postprocess(test_data=ind_data_dict["valid features"])
    for ood_name in ood_names:
        ood_baselines_dict[f"{ood_name} ddu"] = postp.postprocess(test_data=ood_data_dict[f"{ood_name} features"])

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


baseline_name_dict = {
    "pred_h": {
        "plot_title": "Predictive H distribution",
        "x_axis": "Predictive H score",
        "plot_name": "pred_h",
    },
    "mi": {
        "plot_title": "Predictive MI distribution",
        "x_axis": "Predictive MI score",
        "plot_name": "pred_mi",
    },
    "msp": {
        "plot_title": "Predictive MSP distribution",
        "x_axis": "Predictive MSP score",
        "plot_name": "pred_msp",
    },
    "energy": {
        "plot_title": "Predictive energy score distribution",
        "x_axis": "Predictive energy score",
        "plot_name": "pred_energy",
    },
    "mdist": {
        "plot_title": "Mahalanobis Distance distribution",
        "x_axis": "Mahalanobis Distance score",
        "plot_name": "pred_mdist",
    },
    "knn": {
        "plot_title": "kNN distance distribution",
        "x_axis": "kNN Distance score",
        "plot_name": "pred_knn",
    },
    "ash": {
        "plot_title": "ASH score distribution",
        "x_axis": "ASH score",
        "plot_name": "ash_score",
    },
    "dice": {
        "plot_title": "DICE score distribution",
        "x_axis": "DICE score",
        "plot_name": "dice_score",
    },
    "react": {
        "plot_title": "ReAct score distribution",
        "x_axis": "ReAct score",
        "plot_name": "react_score",
    },
    "dice_react": {
        "plot_title": "DICE + ReAct score distribution",
        "x_axis": "DICE + ReAct score",
        "plot_name": "dice_react_score",
    },
    "vim": {
        "plot_title": "ViM score distribution",
        "x_axis": "ViM score",
        "plot_name": "vim_score",
    },
    "gen": {
        "plot_title": "GEN score distribution",
        "x_axis": "GEN score",
        "plot_name": "gen_score",
    },
    "ddu": {
        "plot_title": "DDU score distribution",
        "x_axis": "DDU score",
        "plot_name": "ddu_score",
    },
    "raw": {
        "plot_title": "Raw predictions",
        "x_axis": "Raw predictions",
        "plot_name": "raw_predictions",
    }
}
