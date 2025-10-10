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


# TODO: Add documentation
def get_dice_score_from_features(
    fc_params: Dict[str, np.ndarray],
    ind_data_dict: Dict[str, np.ndarray],
    ood_data_dict: Dict[str, np.ndarray],
    ood_names: List[str],
    ood_baselines_dict: Dict[str, np.ndarray],
    percentile: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
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
    ind_data_dict["dice"] = ind_energy
    # OoD
    for ood_name in ood_names:
        with torch.no_grad():
            ood_logits = (
                dice_layer(Tensor(ood_data_dict[f"{ood_name} features"]).to(device)).cpu().numpy()
            )
        ood_energy = logsumexp(ood_logits, axis=1)
        ood_baselines_dict[f"{ood_name} dice"] = ood_energy

    return ind_data_dict, ood_baselines_dict


def get_react_score_from_features(
    fc_params: Dict[str, np.ndarray],
    ind_data_dict: Dict[str, np.ndarray],
    ood_data_dict: Dict[str, np.ndarray],
    ood_names: List[str],
    ood_baselines_dict: Dict[str, np.ndarray],
    percentile: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
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
    ind_data_dict["react"] = ind_energy
    # OoD
    for ood_name in ood_names:
        ood_clipped_features = ood_data_dict[f"{ood_name} features"].clip(max=activation_threshold)
        ood_logits = np.matmul(ood_clipped_features, w.T) + b
        ood_energy = logsumexp(ood_logits, axis=1)
        ood_baselines_dict[f"{ood_name} react"] = ood_energy

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
    ind_data_dict["dice_react"] = ind_energy
    # OoD
    for ood_name in ood_names:
        ood_clipped_features = ood_data_dict[f"{ood_name} features"].clip(max=activation_threshold)
        with torch.no_grad():
            ood_logits = dice_layer(Tensor(ood_clipped_features).to(device)).cpu().numpy()
        ood_energy = logsumexp(ood_logits, axis=1)
        ood_baselines_dict[f"{ood_name} dice_react"] = ood_energy

    return ind_data_dict, ood_baselines_dict


# ASH with scaling for convolutional layers
def ash_s_conv_layer(x: Tensor, percentile: int = 65):
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
    ind_data_dict["ash"] = ind_energy
    # OoD
    for ood_name in ood_names:
        ood_scattered_features = ash_s_linear_layer(
            ood_data_dict[f"{ood_name} features"], ash_percentile
        )
        ood_logits = np.matmul(ood_scattered_features, w.T) + b
        ood_energy = logsumexp(ood_logits, axis=1)
        ood_baselines_dict[f"{ood_name} ash"] = ood_energy

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
    print("Calculating msp score")
    ind_valid_msp = np.max(softmax(ind_data_dict["valid logits"], axis=1), axis=1)
    ind_data_dict["msp"] = ind_valid_msp
    for ood_name in ood_names:
        ood_msp = np.max(softmax(ood_data_dict[f"{ood_name} logits"], axis=1), axis=1)
        ood_baselines_dict[f"{ood_name} msp"] = ood_msp

    return ind_data_dict, ood_baselines_dict


def get_energy_score_from_logits(
    ind_data_dict: Dict[str, np.ndarray],
    ood_data_dict: Dict[str, np.ndarray],
    ood_names: List[str],
    ood_baselines_dict: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    print("Calculating energy score")
    ind_energy = logsumexp(ind_data_dict["valid logits"], axis=1)
    ind_data_dict["energy"] = ind_energy
    for ood_name in ood_names:
        ood_energy = logsumexp(ood_data_dict[f"{ood_name} logits"], axis=1)
        ood_baselines_dict[f"{ood_name} energy"] = ood_energy

    return ind_data_dict, ood_baselines_dict


def get_mahalanobis_score_from_features(
    ind_data_dict: Dict[str, np.ndarray],
    ood_data_dict: Dict[str, np.ndarray],
    ood_names: List[str],
    ood_baselines_dict: Dict[str, np.ndarray],
    num_classes: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
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
    print("Calculating knn score")
    # Prepare knn model with training data
    train_activations = normalizer(ind_data_dict["train features"])
    index = faiss.IndexFlatL2(ind_data_dict["train features"].shape[1])
    index.add(train_activations)

    def postprocess_knn(data: np.ndarray, k: int) -> np.ndarray:
        k_scores = []
        for sample in data:
            activations = normalizer(sample.reshape(1, -1))
            D, _ = index.search(activations, k)
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

    id_train_logits = id_data.pop("train logits", None) if "train logits" in id_data else None
    id_valid_logits = id_data.pop("valid logits", None) if "valid logits" in id_data else None
    # In the case logits were extracted:
    if isinstance(id_train_logits, np.ndarray) or isinstance(id_valid_logits, np.ndarray):
        if id_train_logits is not None:
            if id_train_logits.shape[1] == 21 or id_train_logits.shape[1] == 11:
                id_train_logits = id_train_logits[:, :-1]
            id_train_labels = np.argmax(id_train_logits, axis=-1)
        else:
            id_train_labels = []
        if id_valid_logits is not None:
            if id_valid_logits.shape[1] == 21 or id_valid_logits.shape[1] == 11:
                id_valid_logits = id_valid_logits[:, :-1]
            id_valid_labels = np.argmax(id_valid_logits, axis=-1)
        else:
            id_valid_labels = []

        id_data["train labels"] = id_train_labels
        id_data["valid labels"] = id_valid_labels
    # In case labels were not extracted
    elif (
        isinstance(id_train_logits, list)
        and len(id_train_logits) == 0
        and isinstance(id_valid_logits, list)
        and len(id_valid_logits) == 0
    ):
        id_data["train labels"] = []
        id_data["valid labels"] = []
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
            ood_data[f"{ood_name} labels"] = []
        else:
            raise NotImplementedError

    return id_data, ood_data


def remove_latent_features(
    id_data: Dict[str, np.ndarray], ood_data: Dict[str, np.ndarray], ood_names: List[str]
):

    id_data.pop("train features", None)
    id_data.pop("valid features", None)
    for ood_name in ood_names:
        ood_data.pop(f"{ood_name} features", None)

    return id_data, ood_data


# DDU calculations
def gmm_fit(
    embeddings: Tensor, labels: Tensor, num_classes: int
) -> Tuple[torch.distributions.MultivariateNormal, float]:
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
    print("Calculating ddu score")
    gmm, _ = gmm_fit(
        embeddings=Tensor(ind_data_dict["train features"]),
        labels=Tensor(ind_data_dict["train labels"]),
        num_classes=num_classes,
    )
    ddu_valid_log_probs = gmm.log_prob(Tensor(ind_data_dict["valid features"][:, None, :]))
    ind_energy = logsumexp(ddu_valid_log_probs.numpy(), axis=1)
    ind_data_dict["ddu"] = ind_energy
    for ood_name in ood_names:
        ood_log_probs = gmm.log_prob(Tensor(ood_data_dict[f"{ood_name} features"][:, None, :]))
        ood_energy = logsumexp(ood_log_probs.numpy(), axis=1)
        ood_baselines_dict[f"{ood_name} ddu"] = ood_energy

    return ind_data_dict, ood_baselines_dict


def get_baselines_thresholds(
    baselines_names: List[str],
    baselines_scores_dict: Dict[str, np.ndarray],
    z_score_percentile: float = 1.645,
) -> Dict[str, float]:
    thresholds = {}
    for baseline_name in baselines_names:
        mean, std = np.mean(baselines_scores_dict[baseline_name]), np.std(
            baselines_scores_dict[baseline_name]
        )
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
